"""
Test 1D/2D/3D Encoder and Decoder with fabricated data.
Tests forward pass, backward pass, and encoder-decoder round-trip shape consistency.
"""
import sys
sys.path.insert(0, '/home/msai/song0304/code/PDE')

import torch


def test_encoder1d():
    """Test PatchifyEncoder1D forward + backward."""
    from pretrain.encoder_v2 import PatchifyEncoder1D

    device = 'cuda'
    dtype = torch.float32

    B, T, X, C = 2, 4, 1024, 3
    print(f"[Encoder1D] Input: [B={B}, T={T}, X={X}, C={C}]")

    encoder = PatchifyEncoder1D(
        in_channels=C,
        hidden_dim=256,
        patch_size=16,
        stem_hidden=64,
        stem_out=128,
        intra_patch_layers=1,
        intra_patch_heads=4,
    ).to(device, dtype)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(B, T, X, C, device=device, dtype=dtype, requires_grad=True)

    tokens, shape_info = encoder(x)
    n_x = X // 16
    expected_shape = (B, T * n_x, 256)
    assert tokens.shape == expected_shape, f"Shape mismatch: {tokens.shape} vs {expected_shape}"
    print(f"  Output: {tokens.shape} OK")
    print(f"  shape_info: {shape_info}")

    loss = tokens.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")


def test_decoder1d():
    """Test PatchifyDecoder1D forward + backward."""
    from pretrain.decoder_v2 import PatchifyDecoder1D

    device = 'cuda'
    dtype = torch.float32

    B, T, n_x, D, C = 2, 4, 64, 256, 3
    X = n_x * 16
    print(f"\n[Decoder1D] Input: [B={B}, T*n_x={T*n_x}, D={D}]")

    decoder = PatchifyDecoder1D(
        out_channels=C,
        hidden_dim=D,
        patch_size=16,
        stem_channels=128,
        decoder_hidden=64,
    ).to(device, dtype)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    tokens = torch.randn(B, T * n_x, D, device=device, dtype=dtype, requires_grad=True)
    shape_info = {'T': T, 'n_x': n_x, 'X': X}

    out = decoder(tokens, shape_info)
    expected_shape = (B, T, X, C)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    print(f"  Output: {out.shape} OK")

    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None
    assert not torch.isnan(tokens.grad).any()
    print(f"  Backward: OK")


def test_encoder2d():
    """Test existing PatchifyEncoder (2D) still works."""
    from pretrain.encoder_v2 import PatchifyEncoder

    device = 'cuda'
    dtype = torch.float32

    B, T, H, W, C = 1, 4, 128, 128, 3
    print(f"\n[Encoder2D] Input: [B={B}, T={T}, H={H}, W={W}, C={C}]")

    encoder = PatchifyEncoder(
        in_channels=C,
        hidden_dim=256,
        patch_size=16,
        stem_hidden=64,
        stem_out=128,
        intra_patch_layers=1,
        intra_patch_heads=4,
    ).to(device, dtype)

    x = torch.randn(B, T, H, W, C, device=device, dtype=dtype, requires_grad=True)

    tokens, shape_info = encoder(x)
    n_h, n_w = H // 16, W // 16
    expected_shape = (B, T * n_h * n_w, 256)
    assert tokens.shape == expected_shape, f"Shape mismatch: {tokens.shape} vs {expected_shape}"
    print(f"  Output: {tokens.shape} OK")

    loss = tokens.sum()
    loss.backward()
    assert x.grad is not None
    print(f"  Backward: OK")


def test_decoder2d():
    """Test existing PatchifyDecoder (2D) still works."""
    from pretrain.decoder_v2 import PatchifyDecoder

    device = 'cuda'
    dtype = torch.float32

    B, T, n_h, n_w, D, C = 1, 4, 8, 8, 256, 3
    H, W = n_h * 16, n_w * 16
    print(f"\n[Decoder2D] Input: [B={B}, T*n_h*n_w={T*n_h*n_w}, D={D}]")

    decoder = PatchifyDecoder(
        out_channels=C,
        hidden_dim=D,
        patch_size=16,
        stem_channels=128,
        decoder_hidden=64,
    ).to(device, dtype)

    tokens = torch.randn(B, T * n_h * n_w, D, device=device, dtype=dtype, requires_grad=True)
    shape_info = {'T': T, 'n_h': n_h, 'n_w': n_w, 'H': H, 'W': W}

    out = decoder(tokens, shape_info)
    expected_shape = (B, T, H, W, C)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    print(f"  Output: {out.shape} OK")

    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None
    print(f"  Backward: OK")


def test_encoder3d():
    """Test PatchifyEncoder3D forward + backward."""
    from pretrain.encoder_v2 import PatchifyEncoder3D

    device = 'cuda'
    dtype = torch.float32

    B, T, D_in, H, W, C = 1, 2, 16, 16, 16, 3
    print(f"\n[Encoder3D] Input: [B={B}, T={T}, D={D_in}, H={H}, W={W}, C={C}]")

    encoder = PatchifyEncoder3D(
        in_channels=C,
        hidden_dim=256,
        patch_size=8,
        stem_hidden=64,
        stem_out=128,
        intra_patch_layers=1,
        intra_patch_heads=4,
    ).to(device, dtype)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(B, T, D_in, H, W, C, device=device, dtype=dtype, requires_grad=True)

    tokens, shape_info = encoder(x)
    n_d, n_h, n_w = D_in // 8, H // 8, W // 8
    expected_shape = (B, T * n_d * n_h * n_w, 256)
    assert tokens.shape == expected_shape, f"Shape mismatch: {tokens.shape} vs {expected_shape}"
    print(f"  Output: {tokens.shape} OK")
    print(f"  shape_info: {shape_info}")

    loss = tokens.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"  Backward: OK")


def test_decoder3d():
    """Test PatchifyDecoder3D forward + backward."""
    from pretrain.decoder_v2 import PatchifyDecoder3D

    device = 'cuda'
    dtype = torch.float32

    B, T = 1, 2
    n_d, n_h, n_w = 2, 2, 2
    D, C, P = 256, 3, 8
    D_in, H, W = n_d * P, n_h * P, n_w * P
    print(f"\n[Decoder3D] Input: [B={B}, T*n_d*n_h*n_w={T*n_d*n_h*n_w}, D={D}]")

    decoder = PatchifyDecoder3D(
        out_channels=C,
        hidden_dim=D,
        patch_size=P,
        stem_channels=128,
        decoder_hidden=64,
    ).to(device, dtype)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    tokens = torch.randn(B, T * n_d * n_h * n_w, D, device=device, dtype=dtype, requires_grad=True)
    shape_info = {'T': T, 'n_d': n_d, 'n_h': n_h, 'n_w': n_w, 'D_in': D_in, 'H': H, 'W': W}

    out = decoder(tokens, shape_info)
    expected_shape = (B, T, D_in, H, W, C)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    print(f"  Output: {out.shape} OK")

    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None
    assert not torch.isnan(tokens.grad).any()
    print(f"  Backward: OK")


def test_roundtrip_1d():
    """Test 1D encoder -> decoder shape consistency."""
    from pretrain.encoder_v2 import PatchifyEncoder1D
    from pretrain.decoder_v2 import PatchifyDecoder1D

    device = 'cuda'
    dtype = torch.float32

    B, T, X, C = 1, 4, 512, 3
    hidden_dim = 256
    print(f"\n[RoundTrip 1D] Input: [B={B}, T={T}, X={X}, C={C}]")

    encoder = PatchifyEncoder1D(
        in_channels=C, hidden_dim=hidden_dim, patch_size=16,
        stem_hidden=64, stem_out=128, intra_patch_layers=1, intra_patch_heads=4,
    ).to(device, dtype)

    decoder = PatchifyDecoder1D(
        out_channels=C, hidden_dim=hidden_dim, patch_size=16,
        stem_channels=128, decoder_hidden=64,
    ).to(device, dtype)

    x = torch.randn(B, T, X, C, device=device, dtype=dtype)
    tokens, shape_info = encoder(x)
    out = decoder(tokens, shape_info)

    assert out.shape == x.shape, f"Round-trip shape mismatch: {out.shape} vs {x.shape}"
    print(f"  {x.shape} -> {tokens.shape} -> {out.shape} OK")


def test_roundtrip_3d():
    """Test 3D encoder -> decoder shape consistency."""
    from pretrain.encoder_v2 import PatchifyEncoder3D
    from pretrain.decoder_v2 import PatchifyDecoder3D

    device = 'cuda'
    dtype = torch.float32

    B, T, D_in, H, W, C = 1, 2, 16, 16, 16, 3
    hidden_dim = 256
    print(f"\n[RoundTrip 3D] Input: [B={B}, T={T}, D={D_in}, H={H}, W={W}, C={C}]")

    encoder = PatchifyEncoder3D(
        in_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_hidden=64, stem_out=128, intra_patch_layers=1, intra_patch_heads=4,
    ).to(device, dtype)

    decoder = PatchifyDecoder3D(
        out_channels=C, hidden_dim=hidden_dim, patch_size=8,
        stem_channels=128, decoder_hidden=64,
    ).to(device, dtype)

    x = torch.randn(B, T, D_in, H, W, C, device=device, dtype=dtype)
    tokens, shape_info = encoder(x)
    out = decoder(tokens, shape_info)

    assert out.shape == x.shape, f"Round-trip shape mismatch: {out.shape} vs {x.shape}"
    print(f"  {x.shape} -> {tokens.shape} -> {out.shape} OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Encoder/Decoder Tests (1D, 2D, 3D)")
    print("=" * 60)

    try:
        test_encoder1d()
        test_decoder1d()
        test_encoder2d()
        test_decoder2d()
        test_encoder3d()
        test_decoder3d()
        test_roundtrip_1d()
        test_roundtrip_3d()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
