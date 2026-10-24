"""Test integracyjny scripts/train.py: krótki trening, komplet plików wynikowych, powtarzalność."""
import csv
import json
from pathlib import Path
import subprocess
import sys

from miner_rl.agents.tabular import TabularAgent


ROOT = Path(__file__).resolve().parents[1]


def run_train(out: Path) -> Path:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--maps", "m1_stabilna",
                    "--seeds", "2", "--episodes", "60", "--q-init", "0", "5", "--eval-every", "30",
                    "--eval-episodes", "5", "--final-episodes", "10", "--workers", "2",
                    "--name", "t", "--out", str(out)], check=True, capture_output=True)
    return out / "t"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_train_outputs_and_reproducibility(tmp_path):
    a = run_train(tmp_path / "a")
    for name in ["config.json", "curves.csv", "final_eval.csv", "runs.csv", "summary.csv"]:
        assert (a / name).exists(), name
    runs = rows(a / "runs.csv")
    assert len(runs) == 2 * 2 * 2                      # algorytmy × q_init × ziarna
    assert len(rows(a / "summary.csv")) == 2 * 2
    assert len(rows(a / "curves.csv")) == 8 * 2        # walidacja po 30 i 60 epizodach
    final = rows(a / "final_eval.csv")
    assert len(final) == 8 * 10
    assert {int(r["seed"]) for r in final} == set(range(1_000_000, 1_000_010))   # ziarna jak baseline
    config = json.loads((a / "config.json").read_text(encoding="utf-8"))
    assert config["args"]["episodes"] == [60] and "m1_stabilna" in config["game_config"]
    model = TabularAgent.load(a / "models" / "m1_stabilna_sarsa_q5_n60_s1.pkl")
    assert model.q_init == 5.0 and model.q

    b = run_train(tmp_path / "b")                      # te same ziarna -> te same wyniki
    strip = lambda rs: [{k: v for k, v in r.items() if k != "train_time_s"} for r in rs]  # noqa: E731
    assert strip(rows(a / "runs.csv")) == strip(rows(b / "runs.csv"))


def test_ofat_variants():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import make_variants, variant_label
    values = {"q_init": [10.0, 30.0], "alpha": [0.1, 0.3, 0.5], "gamma": [0.99],
              "eps_decay": [0.7], "eps_end": [0.05], "episodes": [50_000], "oil_bins": [4],
              "reward_illegal": [-2.0], "reward_collapse": [-10.0], "dig_fraction": [0.3],
              "dig_refund": [0], "mask_illegal": [1], "bonus": [0.0]}
    ofat = make_variants(values, ofat=True)
    assert len(ofat) == 1 + 1 + 2                       # bazowy + zmiany pojedynczych czynników
    assert all(sum(v[k] != ofat[0][k] for k in values) <= 1 for v in ofat)
    assert len(make_variants(values, ofat=False)) == 2 * 3
    assert [variant_label(v) for v in ofat] == ["q10", "q30", "q10_a0.3", "q10_a0.5"]


def test_reward_variants_are_evaluated_with_default_rewards(tmp_path):
    out = tmp_path / "r"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--maps", "m1_stabilna",
                    "--algos", "sarsa", "--seeds", "1", "--episodes", "40", "--ofat",
                    "--reward-collapse", "-10", "-3", "--mask-illegal", "1", "0", "--eval-every", "40",
                    "--eval-episodes", "5", "--final-episodes", "10", "--workers", "2",
                    "--name", "r", "--out", str(out)], check=True, capture_output=True)
    runs = {r["variant"]: r for r in rows(out / "r" / "runs.csv")}
    assert set(runs) == {"q30_n40", "q30_n40_rc-3", "q30_n40_m0"}
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import PARAMS, make_env, train_env
    job = {"map": "m1_stabilna", **{k: v for k, (_, v) in PARAMS.items()}, "reward_collapse": -3.0}
    assert train_env(job).config.reward_collapse == -3.0          # trening: nagrody wariantu
    assert make_env("m1_stabilna").config.reward_collapse == -10.0  # ewaluacja: domyślne nagrody gry
    assert TabularAgent.load(out / "r" / "models" / "m1_stabilna_sarsa_q30_n40_s0.pkl").mask_illegal
    assert not TabularAgent.load(out / "r" / "models" / "m1_stabilna_sarsa_q30_n40_m0_s0.pkl").mask_illegal


def test_resume_skips_finished_runs(tmp_path):
    a = run_train(tmp_path / "a")                      # pełny przebieg (wzorzec)
    out = tmp_path / "b"
    partial = out / "t" / "partial"
    partial.mkdir(parents=True)
    import pickle
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import run
    done = dict(map="m1_stabilna", algo="sarsa", seed=0, episodes=60, q_init=0.0, alpha=0.1, gamma=0.99,
                eps_start=1.0, eps_end=0.05, eps_decay=0.7, oil_bins=4, reward_illegal=-2.0,
                reward_collapse=-10.0, dig_fraction=0.3, dig_refund=0, mask_illegal=1, bonus=0.0,
                eval_every=30, eval_episodes=5, final_episodes=10, eval_seed=1_000_000, out=str(out / "t"),
                variant="q0_n60")
    result = run(done)
    result["train_time"] = -1.0                         # znacznik: wynik wczytany, nie liczony ponownie
    with (partial / "m1_stabilna_sarsa_q0_n60_s0.pkl").open("wb") as f:
        pickle.dump(result, f)
    b = run_train(out)
    runs = rows(b / "runs.csv")
    assert [r["train_time_s"] for r in runs if r["variant"] == "q0_n60" and r["agent"] == "sarsa"
            and r["seed"] == "0"] == ["-1.0"]
    assert not (b / "partial").exists()
    strip = lambda rs: [{k: v for k, v in r.items() if k != "train_time_s"} for r in rs]  # noqa: E731
    assert strip(rows(a / "runs.csv")) == strip(runs)
