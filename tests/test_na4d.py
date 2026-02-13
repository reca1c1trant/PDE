import torch
import sys

def test_na4d_forward_backward():
    """Test 4D Neighborhood Attention forward and backward pass."""
    from natten import NeighborhoodAttention4D

    B, T, D, H, W = 1, 4, 4, 4, 4
    C = 64
    num_heads = 4
    kernel_size = 3

    print(f"[4D] Shape: B={B}, T={T}, D={D}, H={H}, W={W}, C={C}")
    print(f"[4D] num_heads={num_heads}, kernel_size={kernel_size}")

    model = NeighborhoodAttention4D(
        embed_dim=C,
        num_heads=num_heads,
        kernel_size=kernel_size,
        is_causal=(True, False, False, False),
    ).cuda().to(torch.float16)

    x = torch.randn(B, T, D, H, W, C, device='cuda', dtype=torch.float16, requires_grad=True)

    print("[4D] Running forward pass...")
    out = model(x)
    print(f"[4D] Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("[4D] Forward pass OK!")

    print("[4D] Running backward pass...")
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    print("[4D] Backward pass OK!")
    print("[4D] ALL TESTS PASSED!\n")


def test_na3d_still_works():
    """Test 3D Neighborhood Attention still works after modifications."""
    from natten import NeighborhoodAttention3D

    B, T, H, W = 1, 4, 8, 8
    C = 64
    num_heads = 4
    kernel_size = 3

    print(f"[3D] Shape: B={B}, T={T}, H={H}, W={W}, C={C}")
    print(f"[3D] num_heads={num_heads}, kernel_size={kernel_size}")

    model = NeighborhoodAttention3D(
        embed_dim=C,
        num_heads=num_heads,
        kernel_size=kernel_size,
        is_causal=(True, False, False),
    ).cuda().to(torch.float16)

    x = torch.randn(B, T, H, W, C, device='cuda', dtype=torch.float16, requires_grad=True)

    print("[3D] Running forward pass...")
    out = model(x)
    print(f"[3D] Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("[3D] Forward pass OK!")

    print("[3D] Running backward pass...")
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient computed"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    print("[3D] Backward pass OK!")
    print("[3D] ALL TESTS PASSED!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("NATTEN 4D Extension Tests")
    print("=" * 60)

    try:
        test_na3d_still_works()
    except Exception as e:
        print(f"[3D] FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_na4d_forward_backward()
    except Exception as e:
        print(f"[4D] FAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
