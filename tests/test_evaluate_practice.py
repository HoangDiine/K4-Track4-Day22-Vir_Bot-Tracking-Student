from pathlib import Path

from evaluate_practice import _load_eval_config


def test_load_eval_config_infers_benchmark_from_seqinfo_when_config_is_missing(tmp_path: Path) -> None:
    video = tmp_path / "video_1"
    video.mkdir()
    (video / "seqinfo.ini").write_text(
        "[Sequence]\nname=DEMO-01-FRCNN\n",
        encoding="utf-8",
    )

    assert _load_eval_config(tmp_path) == {"benchmark": "DEMO", "split": "train"}
