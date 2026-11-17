"""place_here's scene-camera message conversions (place_container.scene), on stand-in messages: no ROS."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from place_container import scene


def image(array, encoding, pad=0):
    array = np.ascontiguousarray(array)
    rows = [row.tobytes() + b"\0" * pad for row in array]
    return NS(
        width=array.shape[1],
        height=array.shape[0],
        step=array.shape[1] * array.itemsize + pad,
        encoding=encoding,
        is_bigendian=False,
        data=b"".join(rows),
    )


def test_depth_in_millimetres_or_metres_comes_back_in_metres():
    mm = np.array([[0, 1000], [250, 65535]], dtype=np.uint16)
    np.testing.assert_allclose(scene.depth_metres(image(mm, "16UC1", pad=4)), mm / 1000.0, rtol=1e-6)
    m = np.array([[0.5, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(scene.depth_metres(image(m, "32FC1")), [[0.5, 0.0]])
    with pytest.raises(ValueError, match="encoding"):
        scene.depth_metres(image(m, "rgb8"))


def test_the_label_image_is_mask_ids_or_none():
    labels = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint16)
    np.testing.assert_array_equal(scene.label_array(image(labels, "mono16", pad=2)), labels)
    assert scene.label_array(NS(width=0, height=0, step=0, is_bigendian=False, data=b"")) is None


def test_a_transform_becomes_a_matrix():
    s = np.sqrt(0.5)  # 90 deg about z
    tf = NS(translation=NS(x=1.0, y=2.0, z=3.0), rotation=NS(x=0.0, y=0.0, z=s, w=s))
    T = scene.matrix(tf)
    np.testing.assert_allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
