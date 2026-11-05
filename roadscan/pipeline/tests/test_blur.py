import numpy as np
import pytest

from roadscan.blur import pad_boxes, pixelate

cv2 = pytest.importorskip("cv2")


def test_pad_boxes_grows_and_clips():
    boxes = pad_boxes([[10, 10, 30, 50], [90, 0, 100, 10]], pad=0.5, width=100, height=60)
    assert boxes.tolist() == [[0, 0, 40, 60], [85, 0, 100, 15]]
    assert pad_boxes(np.zeros((0, 4)), 0.1, 100, 100).shape == (0, 4)


def test_pixelate_changes_only_inside_boxes():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    original = image.copy()
    pixelate(image, np.array([[40, 30, 120, 90]]))
    inside, before = image[30:90, 40:120], original[30:90, 40:120]
    assert np.mean(inside != before) > 0.9  # detail destroyed
    assert len(np.unique(inside.reshape(-1, 3), axis=0)) <= 12 * 12  # coarse cells only
    mask = np.ones(image.shape[:2], bool)
    mask[30:90, 40:120] = False
    assert np.array_equal(image[mask], original[mask])
