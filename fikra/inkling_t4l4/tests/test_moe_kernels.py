"""CPU-only checks for the MXFP8 expert format and the SiTU activation.

The point of these is that they can fail here, without a GPU, so a Kaggle cycle
is never spent discovering an arithmetic mistake in the activation.
"""
import numpy as np
import pytest

from fikra.inkling_t4l4 import moe_kernels as mk
from fikra.quant.quant_ref import e4m3_to_float


def test_self_check_passes():
    mk._self_check()


def test_e4m3_reference_values():
    # 1.0, 2.0, -1.0 in E4M3 (bias 7, three mantissa bits)
    got = e4m3_to_float(np.array([0x38, 0x40, 0xB8], np.uint8))
    assert list(got) == [1.0, 2.0, -1.0]


def test_ue8m0_is_a_power_of_two():
    from fikra.inkling_t4l4.nvfp4 import ue8m0_to_float
    assert ue8m0_to_float(bytes([127]))[0] == 1.0
    assert ue8m0_to_float(bytes([134]))[0] == 128.0
    assert ue8m0_to_float(bytes([118]))[0] == pytest.approx(2.0 ** -9)


def test_mxfp8_gemv_shape_contract():
    """The fallback path must at least agree with a hand GEMV on CPU."""
    torch = pytest.importorskip("torch")
    n, k = 64, 64
    codes = torch.zeros(2, n, k, dtype=torch.uint8)
    codes[0, 0, 0] = 0x38          # 1.0 in the first cell of expert 0
    scales = torch.full((2, n, k // 32), 127, dtype=torch.uint8)   # 2**0
    x = torch.zeros(k, dtype=torch.float32)
    x[0] = 3.0
    out = mk.mxfp8_gemv(x, codes, scales, torch.tensor([0]), n, k)
    assert out.shape == (1, n)
    assert float(out[0, 0]) == pytest.approx(3.0)
    assert float(out[0, 1]) == 0.0


def test_situ_differs_from_plain_swiglu():
    """The failure mode this guards: silu(gate)*up looks plausible and is wrong."""
    torch = pytest.importorskip("torch")
    gate_up = torch.tensor([[2.0, 1.0]])
    situ = float(mk.capped_swiglu(gate_up, 4.0, 25.0)[0, 0])
    plain = float(torch.nn.functional.silu(torch.tensor(2.0)) * 1.0)
    assert situ != pytest.approx(plain, rel=1e-3)
