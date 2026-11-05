"""End-to-end on synthetic rides. Thresholds guard against regressions; they are not real-world accuracy."""
import pytest

from roadscan.__main__ import main
from roadscan.evaluate import evaluate
from roadscan.synth import generate


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    rides = generate(tmp_path_factory.mktemp("rides"), rides=8, repair_after=3, seed=7)
    return evaluate(rides), rides


def test_fusion_detection(result):
    totals = result[0]["totals"]
    fusion = totals.loc["fusion"]
    assert fusion["precision"] >= 0.8
    assert fusion["recall"] >= 0.65
    assert fusion["median_err_m"] < 5
    # Never worse than the best single vision threshold on F1.
    assert fusion["f1"] >= totals.drop("fusion")["f1"].max()


def test_registry_dedups_and_notices_repairs(result):
    r = result[0]["registry"]
    assert r["found"] == r["true_potholes"]
    assert r["duplicates"] <= 0.15 * r["true_potholes"]
    assert r["median_err_m"] < 3
    assert r["repair_noticed"] >= 0.75 * r["repaired"]
    assert r["open_wrongly_closed"] == 0


def test_cli_process_and_map(result, tmp_path):
    rides = result[1]
    db = tmp_path / "cli.db"
    sessions = [str(rides / f"ride_{i:02d}") for i in (1, 2)]
    assert main(["process", *sessions, "--db", str(db)]) == 0
    assert main(["process", sessions[0], "--db", str(db)]) == 1  # already ingested
    out = tmp_path / "map.html"
    assert main(["map", "--db", str(db), "--out", str(out), "--exports", str(tmp_path / "ex")]) == 0
    html = out.read_text(encoding="utf-8")
    assert '"type": "FeatureCollection"' in html and "__DATA__" not in html
    assert (tmp_path / "ex" / "potholes.geojson").exists()


def test_load_config_overrides_and_rejects_typos(tmp_path):
    from roadscan.fusion import FusionConfig, load_config

    good = tmp_path / "model.fusion.json"
    good.write_text('{"_note": "comment", "high_conf": 0.4, "medium_conf": 0.15}')
    cfg = load_config(good)
    assert (cfg.high_conf, cfg.medium_conf, cfg.crack_conf) == (0.4, 0.15, FusionConfig().crack_conf)

    bad = tmp_path / "typo.json"
    bad.write_text('{"hgh_conf": 0.4}')
    with pytest.raises(ValueError, match="hgh_conf"):
        load_config(bad)


def test_shipped_model_config_loads():
    from pathlib import Path

    from roadscan.fusion import load_config

    path = Path(__file__).resolve().parents[3] / "models" / "roadscan_rdd_india_y11m.fusion.json"
    if not path.exists():
        pytest.skip("model config not present")
    assert load_config(path).high_conf == 0.4
