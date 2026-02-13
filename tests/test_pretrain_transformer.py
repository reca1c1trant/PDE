"""
Test pretrain NATransformer with fabricated data.
Only tests the transformer (NA layers), NOT encoder/decoder.
"""
import sys
sys.path.insert(0, '/home/msai/song0304/code/PDE')

import torch
from pretrain.attention_v2 import NATransformerLayer, NeighborhoodAttention3D, RMSNorm
from pretrain.model_v2 import NATransformer


def test_na_transformer_layer():
    """Test single NATransformerLayer with fake tokens."""
    device = 'cuda'
    dtype = torch.float16

    B, T, n_h, n_w, D = 1, 4, 4, 4, 128
    num_heads = 4
    kernel_size = (3, 3, 3)

    print(f"[NATransformerLayer] Input: [B={B}, T={T}, n_h={n_h}, n_w={n_w}, D={D}]")
    print(f"  num_heads={num_heads}, kernel_size={kernel_size}")

    layer = NATransformerLayer(
        hidden_dim=D,
        num_heads=num_heads,
        kernel_size=kernel_size,
        is_causal=True,
        dropout=0.0,
    ).to(device, dtype)

    x = torch.randn(B, T, n_h, n_w, D, device=device, dtype=dtype, requires_grad=True)

    out = layer(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Output shape: {out.shape} OK")

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")


def test_na_transformer_full():
    """Test the full NATransformer stack with fabricated tokens."""
    device = 'cuda'
    dtype = torch.float16

    # Simulate encoder output: small model
    B = 1
    T = 4        # 4 time steps
    n_h, n_w = 4, 4   # 4x4 patch grid (e.g., 64x64 image with patch_size=16)
    D = 128      # hidden dim
    num_layers = 2
    num_heads = 4
    base_kernel = 3

    print(f"\n[NATransformer] Config: layers={num_layers}, heads={num_heads}, kernel={base_kernel}")
    print(f"  Simulated encoder output: B={B}, T={T}, n_h={n_h}, n_w={n_w}, D={D}")
    print(f"  Token sequence length: T*n_h*n_w = {T*n_h*n_w}")

    transformer = NATransformer(
        hidden_dim=D,
        num_layers=num_layers,
        num_heads=num_heads,
        base_kernel=base_kernel,
        dropout=0.0,
    ).to(device, dtype)

    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  Parameters: {n_params:,}")

    # Fabricate tokens as if coming from encoder
    tokens = torch.randn(B, T * n_h * n_w, D, device=device, dtype=dtype, requires_grad=True)
    shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w}

    print(f"  Forward pass...")
    out = transformer(tokens, shape_info)
    assert out.shape == tokens.shape, f"Shape mismatch: {out.shape} vs {tokens.shape}"
    print(f"  Output: {out.shape} OK")

    print(f"  Backward pass...")
    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None
    assert not torch.isnan(tokens.grad).any()
    print(f"  Backward: OK")


def test_na_transformer_aspect_ratios():
    """Test NATransformer with different aspect ratios (1:1, 1:2, 1:4)."""
    device = 'cuda'
    dtype = torch.float16

    D = 128
    num_layers = 2
    num_heads = 4
    base_kernel = 3

    transformer = NATransformer(
        hidden_dim=D,
        num_layers=num_layers,
        num_heads=num_heads,
        base_kernel=base_kernel,
        dropout=0.0,
    ).to(device, dtype)

    test_cases = [
        # (T, n_h, n_w, expected_layer_set)
        (4, 4, 4,  "1x1"),  # square
        (4, 4, 8,  "1x2"),  # 1:2 ratio
        (4, 4, 16, "1x4"),  # 1:4 ratio
    ]

    print(f"\n[Aspect Ratio Tests]")
    for T, n_h, n_w, label in test_cases:
        B = 1
        tokens = torch.randn(B, T * n_h * n_w, D, device=device, dtype=dtype)
        shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w}

        out = transformer(tokens, shape_info)
        assert out.shape == tokens.shape
        print(f"  {label} (n_h={n_h}, n_w={n_w}): {tokens.shape} -> {out.shape} OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Pretrain NATransformer Tests (no encoder/decoder)")
    print("=" * 60)

    try:
        test_na_transformer_layer()
        test_na_transformer_full()
        test_na_transformer_aspect_ratios()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
