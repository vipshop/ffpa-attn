"""Compile the SM100 exp2 helper without requiring an SM100 device."""

import cutlass
import cutlass.cute as cute
import pytest
import torch

from ffpa_attn.cute.utils.softmax import ex2_emulation_2


@cute.kernel
def _ex2_kernel(inputs: cute.Tensor, outputs: cute.Tensor):
  i = cute.arch.thread_idx()[0] * 2
  x, y = ex2_emulation_2(inputs[i], inputs[i + 1])
  outputs[i] = x
  outputs[i + 1] = y


@cute.jit
def _ex2(inputs: cute.Tensor, outputs: cute.Tensor):
  _ex2_kernel(inputs, outputs).launch(grid=(1, 1, 1), block=(32, 1, 1))


def test_ex2_emulation_compiles_for_sm100():
  """Trace the real helper so removed dependency APIs fail on any host."""
  tensor = cute.runtime.make_fake_tensor(
    cutlass.Float32, (64, ), stride=(1, ), assumed_align=16
  )
  assert cute.compile(
    _ex2, tensor, tensor, options="--gpu-arch sm_100a"
  ) is not None


@pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
  reason="packed f32x2 exp2 requires SM100 or newer",
)
def test_ex2_emulation_matches_exp2():
  inputs = torch.linspace(-80.0, 0.0, 64, device="cuda")
  outputs = torch.empty_like(inputs)
  mInputs = cute.runtime.from_dlpack(inputs)
  mOutputs = cute.runtime.from_dlpack(outputs)
  kernel = cute.compile(_ex2, mInputs, mOutputs)
  kernel(mInputs, mOutputs)
  torch.testing.assert_close(outputs, inputs.exp2(), rtol=1e-3, atol=0)
