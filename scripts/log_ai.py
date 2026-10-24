#!/usr/bin/env python3
"""Append a complete AI exchange to the project's public LaTeX journal (ai-journal/).

Only Python's standard library is required. This script does not capture chats
automatically, redact secrets, compile TeX, or create the journal's main.tex.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
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
DEFAULT_VERIFICATION = "Do uzupełnienia."


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


def last_sunday_utc(year: int, month: int) -> datetime:
    """Ostatnia niedziela miesiąca, godz. 01:00 UTC (moment zmiany czasu w UE)."""
    day = datetime(year, month + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
    return day - timedelta(days=(day.weekday() + 1) % 7)


def warsaw_offset(now_utc: datetime) -> timedelta:
    """Przesunięcie Europe/Warsaw według reguł UE, gdy brak bazy tzdata (np. Windows)."""
    year = now_utc.year
    summer = last_sunday_utc(year, 3) <= now_utc < last_sunday_utc(year, 10)
    return timedelta(hours=2 if summer else 1)


def warsaw_date(now_utc: datetime | None = None) -> str:
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        return now_utc.astimezone(ZoneInfo("Europe/Warsaw")).date().isoformat()
    except ZoneInfoNotFoundError:
        return (now_utc + warsaw_offset(now_utc)).date().isoformat()


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


@contextmanager
def entry_staging(entries_dir: Path):
    """Use inherited Windows ACLs, unlike TemporaryDirectory's private ACLs."""
    staging_name = ".entry-" + uuid4().hex
    staging = entries_dir / staging_name
    staging.mkdir()
    try:
        yield staging
    finally:
        # After the successful rename this path no longer exists. On failure,
        # remove only the four files this operation could have created. Never
        # recursively delete an unchecked or unexpectedly populated directory.
        if staging.exists():
            if (
                staging.is_symlink()
                or staging.name != staging_name
                or not re.fullmatch(r"\.entry-[0-9a-f]{32}", staging.name)
                or staging.resolve().parent != entries_dir.resolve()
            ):
                raise ValueError("Odmowa sprzątania nieoczekiwanej ścieżki stagingu.")
            for filename in ("prompt.txt", "response.txt", "metadata.json", "entry.tex"):
                (staging / filename).unlink(missing_ok=True)
            staging.rmdir()


def entry_tex(entry_id: str, metadata: dict[str, str]) -> str:
    tool = metadata.get("tool", "")
    if metadata.get("model"):
        tool = f"{tool} ({metadata['model']})" if tool else metadata["model"]
    rows = (
        ("Data", metadata.get("date", "")),
        ("Narzędzie", tool),
        ("Cel", metadata.get("purpose", "")),
        ("Zakres pomocy", metadata.get("scope", "")),
        ("Weryfikacja", metadata.get("verification", "")),
        ("Link do rozmowy", metadata.get("conversation_url", "")),
    )
    title = f"{metadata.get('date', '')} -- {metadata.get('purpose', '')}".strip(" -")
    lines = [rf"\section{{{latex_escape(title)}}}", rf"\label{{{entry_id}}}"]
    for label, value in rows:
        if value:
            lines.append(rf"\metawiersz{{{label}}}{{{latex_escape(value)}}}")
    lines.extend(
        [
            r"\subsection*{Prompt}",
            rf"\VerbatimInput{{entries/{entry_id}/prompt.txt}}",
            r"\subsection*{Odpowiedź}",
            rf"\VerbatimInput{{entries/{entry_id}/response.txt}}",
            r"\bigskip\hrule\bigskip",
            "",
        ]
    )
    return "\n".join(lines)


def rebuild_entries(journal_dir: Path) -> int:
    """Regenerate entry.tex for every entry from its metadata (after a format change)."""
    count = 0
    for directory in (journal_dir / "entries").iterdir():
        if directory.is_dir() and not directory.is_symlink() and ENTRY_PATTERN.fullmatch(directory.name):
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            (directory / "entry.tex").write_text(entry_tex(directory.name, metadata), encoding="utf-8")
            count += 1
    rebuild_index(journal_dir)
    return count


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
        with entry_staging(entries_dir) as staging:
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
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--response-file", type=Path)
    parser.add_argument("--model", help="Rzeczywista nazwa modelu; nie zgaduj wariantu.")
    parser.add_argument("--tool", default="Codex")
    parser.add_argument("--purpose")
    parser.add_argument("--scope")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Przebuduj entry.tex wszystkich wpisów (np. po zmianie formatu) i zakończ.",
    )
    parser.add_argument("--date", type=parse_date, help="YYYY-MM-DD; domyślnie Europe/Warsaw.")
    parser.add_argument("--verification", default=DEFAULT_VERIFICATION)
    parser.add_argument("--conversation-url")
    parser.add_argument(
        "--journal-dir", type=Path, default=PROJECT_ROOT / "ai-journal",
        help="Katalog dziennika; zmień tylko świadomie (np. na potrzeby testu).",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.rebuild:
        count = rebuild_entries(args.journal_dir.resolve())
        print(f"Przebudowano wpisów: {count}")
        return 0
    missing = [name for name in ("prompt_file", "response_file", "model", "purpose", "scope")
               if getattr(args, name) is None]
    if missing:
        parser.error("brak wymaganych argumentów: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    try:
        destination = append_entry(args)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"Błąd zapisu dziennika AI: {error}\n")
    print(f"Zapisano wpis AI: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
