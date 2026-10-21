#!/usr/bin/env python3
"""Append a complete AI exchange to the local, private LaTeX journal.

Only Python's standard library is required. This script does not capture chats
automatically, redact secrets, compile TeX, or create the journal's main.tex.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY_PATTERN = re.compile(r"\d{8}-[0-9a-f]{12}\Z")
DEFAULT_VERIFICATION = (
    "Niepotwierdzona: student powinien samodzielnie sprawdzić i zrozumieć "
    "wykorzystaną pomoc."
)


def latex_escape(value: str) -> str:
    """Escape metadata as text; raw conversations are read with VerbatimInput."""
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\n": r"\par{} ",
        "\r": "",
        "\t": " ",
    }
    return "".join(replacements.get(character, character) for character in value)


def parse_date(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("Data musi mieć format YYYY-MM-DD.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Niepoprawna data kalendarzowa.") from error


def warsaw_date() -> str:
    try:
        return datetime.now(ZoneInfo("Europe/Warsaw")).date().isoformat()
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            "Brak systemowej bazy stref czasowych Europe/Warsaw; podaj --date YYYY-MM-DD."
        ) from error


def read_utf8(path: Path) -> bytes:
    # Preserve the original bytes, including whitespace, BOM and line endings.
    content = path.read_bytes()
    content.decode("utf-8")
    if not content:
        raise ValueError(f"Plik nie może być pusty: {path}")
    return content


@contextmanager
def journal_lock(journal_dir: Path):
    """Serialize concurrent agents so rebuilding entries.tex cannot lose entries."""
    lock_path = journal_dir / ".append.lock"
    deadline = time.monotonic() + 10
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ValueError(
                    "Dziennik jest zablokowany. Sprawdź, czy inny agent zapisuje wpis. "
                    "Osieroconą blokadę .append.lock usuń dopiero po sprawdzeniu."
                )
            time.sleep(0.1)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def entry_tex(entry_id: str, metadata: dict[str, str]) -> str:
    labels = (
        ("date", "Data użycia"),
        ("tool", "Narzędzie"),
        ("model", "Model"),
        ("purpose", "Cel użycia"),
        ("scope", "Zakres wykorzystanej pomocy"),
        ("verification", "Samodzielna weryfikacja studenta"),
        ("conversation_url", "Link do pełnej rozmowy"),
    )
    lines = [
        rf"\section{{Wymiana {latex_escape(entry_id)}}}",
        r"\begin{description}",
    ]
    for key, label in labels:
        if metadata.get(key):
            lines.append(rf"\item[{label}] {latex_escape(metadata[key])}")
    lines.extend(
        [
            r"\end{description}",
            r"\subsection*{Prompt}",
            rf"\VerbatimInput[breaklines=true,breakanywhere=true,fontsize=\small]{{entries/{entry_id}/prompt.txt}}",
            r"\subsection*{Odpowiedź}",
            rf"\VerbatimInput[breaklines=true,breakanywhere=true,fontsize=\small]{{entries/{entry_id}/response.txt}}",
            "",
        ]
    )
    return "\n".join(lines)


def rebuild_index(journal_dir: Path) -> None:
    directories = [
        directory
        for directory in (journal_dir / "entries").iterdir()
        if not directory.is_symlink()
        and directory.is_dir()
        and ENTRY_PATTERN.fullmatch(directory.name)
        and (directory / "entry.tex").is_file()
    ]
    def sort_key(directory: Path) -> tuple[str, str, str]:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return metadata["date"], metadata.get("recorded_at_utc", ""), directory.name

    entry_ids = [directory.name for directory in sorted(directories, key=sort_key)]
    content = "% Generowane przez scripts/log_ai.py; nie edytuj ręcznie.\n"
    content += "".join(rf"\input{{entries/{entry_id}/entry.tex}}" + "\n" for entry_id in entry_ids)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=journal_dir,
        prefix=".entries-", suffix=".tmp", delete=False,
    ) as index_file:
        temporary_path = Path(index_file.name)
        index_file.write(content)
    try:
        os.replace(temporary_path, journal_dir / "entries.tex")
    finally:
        temporary_path.unlink(missing_ok=True)


def append_entry(args: argparse.Namespace) -> Path:
    prompt = read_utf8(args.prompt_file)
    response = read_utf8(args.response_file)
    used_date = args.date or warsaw_date()
    if args.conversation_url:
        url = urlsplit(args.conversation_url)
        if url.scheme not in {"https", "http"} or not url.netloc or url.username or url.password:
            raise ValueError("Link rozmowy musi być adresem HTTP(S) bez danych logowania.")
    journal_dir = args.journal_dir.resolve()
    journal_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "date": used_date,
        "tool": args.tool,
        "model": args.model,
        "purpose": args.purpose,
        "scope": args.scope,
        "verification": args.verification,
        "conversation_url": args.conversation_url or "",
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "response_sha256": hashlib.sha256(response).hexdigest(),
    }
    with journal_lock(journal_dir):
        metadata["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        entries_dir = journal_dir / "entries"
        entries_dir.mkdir(exist_ok=True)
        entry_id = used_date.replace("-", "") + "-" + uuid4().hex[:12]
        destination = entries_dir / entry_id
        with tempfile.TemporaryDirectory(prefix=".entry-", dir=entries_dir) as temporary_dir:
            staging = Path(temporary_dir)
            (staging / "prompt.txt").write_bytes(prompt)
            (staging / "response.txt").write_bytes(response)
            (staging / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (staging / "entry.tex").write_text(entry_tex(entry_id, metadata), encoding="utf-8")
            os.replace(staging, destination)
        rebuild_index(journal_dir)
    return destination


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--response-file", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Rzeczywista nazwa modelu; nie zgaduj wariantu.")
    parser.add_argument("--tool", default="Codex")
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--date", type=parse_date, help="YYYY-MM-DD; domyślnie Europe/Warsaw.")
    parser.add_argument("--verification", default=DEFAULT_VERIFICATION)
    parser.add_argument("--conversation-url")
    parser.add_argument(
        "--journal-dir", type=Path, default=PROJECT_ROOT / "private" / "ai-journal",
        help="Katalog lokalnego dziennika; zmień tylko świadomie (np. na potrzeby testu).",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        destination = append_entry(args)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Błąd zapisu dziennika AI: {error}\n")
    print(f"Zapisano lokalny wpis AI: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
