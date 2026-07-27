from types import SimpleNamespace

import cv2
import numpy as np

from aquaadapt.bag.image_decoder import decode_compressed_image, decode_raw_image


def test_raw_rgb_to_bgr_and_row_padding() -> None:
    rgb_rows = [
        bytes([255, 0, 0, 0, 255, 0, 99, 99]),
        bytes([0, 0, 255, 255, 255, 255, 88, 88]),
    ]
    message = SimpleNamespace(
        encoding="rgb8", height=2, width=2, step=8, is_bigendian=0,
        data=b"".join(rgb_rows),
    )
    decoded = decode_raw_image(message).image
    assert decoded.shape == (2, 2, 3)
    assert decoded[0, 0].tolist() == [0, 0, 255]
    assert decoded[1, 0].tolist() == [255, 0, 0]


def test_raw_big_endian_mono16() -> None:
    data = np.array([[0, 1000], [2000, 4000]], dtype=">u2").tobytes()
    message = SimpleNamespace(encoding="mono16", height=2, width=2, step=4, is_bigendian=1, data=data)
    decoded = decode_raw_image(message).image
    assert decoded.dtype == np.uint8
    assert decoded.shape == (2, 2)


def test_compressed_decode() -> None:
    source = np.zeros((8, 9, 3), np.uint8)
    source[:, :, 1] = 180
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    decoded = decode_compressed_image(SimpleNamespace(data=encoded.tobytes(), format="png")).image
    assert np.array_equal(decoded, source)

