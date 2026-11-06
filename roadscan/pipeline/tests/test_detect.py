from types import SimpleNamespace

from roadscan.detect import trained_imgsz


def test_trained_imgsz_reads_checkpoint_args():
    assert trained_imgsz(SimpleNamespace(ckpt={"train_args": {"imgsz": 736}})) == 736
    assert trained_imgsz(SimpleNamespace(ckpt={"train_args": {"imgsz": [736, 736]}})) == 736


def test_trained_imgsz_falls_back_without_checkpoint():
    assert trained_imgsz(SimpleNamespace(ckpt=None)) == 640
    assert trained_imgsz(SimpleNamespace()) == 640
