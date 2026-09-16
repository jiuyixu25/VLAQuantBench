import numpy as np

from vlaquantbench.remote import decode, encode


def test_roundtrip_nested():
    obj = {
        "img": np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "tuple": (np.zeros(3), np.array([0, 0, 0, 1.0]), -1),
        "nested": {"a": [1, 2.5, "x", None], "mat": np.eye(3)},
        "dropped": object(),
    }
    out = decode(encode(obj))
    assert np.array_equal(out["img"], obj["img"]) and out["img"].dtype == np.uint8
    assert np.array_equal(out["state"], obj["state"]) and out["state"].dtype == np.float32
    assert isinstance(out["tuple"], tuple) and out["tuple"][2] == -1
    assert out["nested"]["a"] == [1, 2.5, "x", None] and np.array_equal(out["nested"]["mat"], np.eye(3))
    assert "dropped" not in out
