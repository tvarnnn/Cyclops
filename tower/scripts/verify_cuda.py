"""One-shot PyTorch/CUDA verification for this Tower machine.

Checks that torch imports, reports its CUDA runtime, device and capability,
and runs a tiny matmul on the GPU. Exits non-zero when CUDA is unavailable.
Run manually:
    .venv\\Scripts\\python.exe scripts/verify_cuda.py

Not part of the pytest suite -- this needs a real GPU and a real torch
install, neither of which the fast test suite may assume.
"""

import torch


def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"cuda available: {available}")
    if not available:
        raise SystemExit(
            "CUDA not available to torch -- STOP. Escalate before "
            "implementing any CUDA-backed experiment."
        )

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"device: {device_name}")
    print(f"capability: {capability}")

    a = torch.rand(4, 4, device="cuda")
    b = torch.rand(4, 4, device="cuda")
    result = (a @ b).sum().item()
    print(f"trivial matmul sum on cuda: {result}")


if __name__ == "__main__":
    main()
