import numpy as np

from abstractvoice.audio.fade import apply_edge_fades
from abstractvoice.audio.fade import ensure_headroom


def test_apply_edge_fades_zeroes_edges():
    sr = 24000
    x = np.ones((1000,), dtype=np.float32)
    y = apply_edge_fades(x, sample_rate=sr, fade_ms=5.0, fade_in=True, fade_out=True)
    assert y.shape == x.shape
    assert float(y[0]) == 0.0
    assert float(y[-1]) == 0.0


def test_apply_edge_fades_reduces_boundary_jump_between_chunks():
    sr = 24000
    a = np.ones((1000,), dtype=np.float32)
    b = -np.ones((1000,), dtype=np.float32)
    a2 = apply_edge_fades(a, sample_rate=sr, fade_ms=5.0, fade_in=False, fade_out=True)
    b2 = apply_edge_fades(b, sample_rate=sr, fade_ms=5.0, fade_in=True, fade_out=False)
    # If we stitch a2 then b2, the discontinuity at the boundary is minimized.
    boundary_jump = abs(float(b2[0] - a2[-1]))
    assert boundary_jump <= 0.01


def test_ensure_headroom_scales_only_when_needed():
    x = np.array([0.5, -0.5], dtype=np.float32)
    y = ensure_headroom(x, headroom=0.98)
    assert np.allclose(x, y)

    x2 = np.array([1.2, -1.2], dtype=np.float32)
    y2 = ensure_headroom(x2, headroom=0.98)
    assert float(np.max(np.abs(y2))) <= 0.980001

