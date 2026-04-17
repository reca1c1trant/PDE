"""
Muon-LoRA: LoRA with Newton-Schulz Orthogonalization in Forward Path.

Key idea: G = BA, alpha = ||G||_F, M = NS(G), ΔW = alpha * M
- NS orthogonalizes G → UV^T (nearest semi-orthogonal matrix)
- alpha = ||BA|| preserves original scale, grows naturally during training
- Orthogonal update preserves pretrain feature space structure (rotation/reflection only)
- NS is fully differentiable → gradients flow to A, B via autograd

Drop-in replacement for model_lora_v3.py with same interface.

Usage:
    from finetune.model_lora_muon import PDELoRAModelMuon, save_lora_checkpoint, load_lora_checkpoint
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pretrain.model_v3 import PDEModelV3


def _is_main_process() -> bool:
    return os.environ.get('LOCAL_RANK', '0') == '0'


# ============================================================
# Newton-Schulz Orthogonalization
# ============================================================

def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration: orthogonalize G → nearest semi-orthogonal UV^T.

    For G = USV^T, converges singular values S → 1, preserving U and V.
    Rank-preserving: rank-r input → rank-r output with all non-zero σ ≈ 1.
    Fully differentiable (all ops are matmul/add).

    Quintic polynomial φ(x) = 3.4445x - 4.7750x³ + 2.0315x⁵
    Applied iteratively: σ → φ^N(σ) → 1 for σ ∈ (0, 1].
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G / (G.norm() + eps)

    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


# ============================================================
# Muon-LoRA Linear Layer
# ============================================================

class MuonLoRALinear(nn.Module):
    """
    Linear layer with Muon-LoRA adaptation.

    Forward: G = BA, alpha = ||G||, M = NS(G), out = W_base·x + alpha·M·x

    alpha = ||BA||_F preserves the natural scale of the low-rank update.
    NS(BA) orthogonalizes → all r directions contribute equally.
    B initialized small → alpha ≈ 0 at start → near-identity init.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_dropout: float = 0.05,
        ns_steps: int = 5,
        init_scale: float = 0.01,
        use_gate: bool = False,
    ):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.ns_steps = ns_steps
        self.use_gate = use_gate

        # A: orthogonal init (well-conditioned rows for NS)
        # B: dimension-aware init so ||BA||_F ≈ init_scale across all layers
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.orthogonal_(self.lora_A)
        init_std = init_scale / math.sqrt(out_features * r)
        nn.init.normal_(self.lora_B, std=init_std)

        # Gate: init=0 → ΔW=0 at step 0 (exact pretrained start)
        if use_gate:
            self.lora_gate = nn.Parameter(torch.zeros(1))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        G = self.lora_B @ self.lora_A
        alpha = G.norm()
        M = newtonschulz5(G, steps=self.ns_steps)
        lora_out = F.linear(self.lora_dropout(x), M)

        if self.use_gate:
            return base_out + self.lora_gate * alpha * lora_out
        return base_out + alpha * lora_out

    def extra_repr(self) -> str:
        gate_str = ", gate=True" if self.use_gate else ""
        return (f"in={self.base_linear.in_features}, out={self.base_linear.out_features}, "
                f"r={self.r}, ns_steps={self.ns_steps}{gate_str}")


# ============================================================
# Split-NS Orthogonal LoRA Linear Layer
# ============================================================

class SplitNSLoRALinear(nn.Module):
    """
    Orthogonal LoRA with separate NS on A and B, scaled by their norms.

    Forward: ΔW = ||A|| * ||B|| * NS(B) @ NS(A)
    - NS(A): rows orthonormal [r, in]
    - NS(B): columns orthonormal [out, r]
    - ||A||, ||B||: preserve original scale of each matrix independently
    - A, B loaded from trained vanilla LoRA → non-zero init, no gate needed
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_dropout: float = 0.05,
        ns_steps: int = 5,
        init_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.ns_steps = ns_steps

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.orthogonal_(self.lora_A)
        init_std = init_scale / math.sqrt(out_features * r)
        nn.init.normal_(self.lora_B, std=init_std)

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        a = self.lora_A.norm()
        b = self.lora_B.norm()
        A_orth = newtonschulz5(self.lora_A, steps=self.ns_steps)
        B_orth = newtonschulz5(self.lora_B, steps=self.ns_steps)

        h = F.linear(self.lora_dropout(x), A_orth)   # [batch, r]
        h = F.linear(h, B_orth)                        # [batch, out]

        return base_out + a * b * h

    def extra_repr(self) -> str:
        return (f"in={self.base_linear.in_features}, out={self.base_linear.out_features}, "
                f"r={self.r}, ns_steps={self.ns_steps}")


# ============================================================
# Multi-Head Orthogonal LoRA Linear Layer
# ============================================================

class MultiHeadMuonLoRALinear(nn.Module):
    """
    Multi-head orthogonal LoRA: split rank-r into k heads, each independently NS'd.

    ΔW = Σ_h alpha_h * NS(B_h @ A_h)   (h=1..n_heads, each rank r/n_heads)

    Each head has independent alpha → different feature subspaces get different
    update magnitudes. Within each head: orthogonal (no rank collapse).
    Same A [r, in] and B [out, r] as MuonLoRALinear — no extra parameters.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_dropout: float = 0.05,
        ns_steps: int = 5,
        init_scale: float = 0.01,
        n_heads: int = 4,
    ):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        assert r % n_heads == 0, f"r={r} must be divisible by n_heads={n_heads}"

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.ns_steps = ns_steps
        self.n_heads = n_heads
        self.head_r = r // n_heads

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.orthogonal_(self.lora_A)
        init_std = init_scale / math.sqrt(out_features * r)
        nn.init.normal_(self.lora_B, std=init_std)

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        x_drop = self.lora_dropout(x)

        hr = self.head_r
        lora_out = torch.zeros_like(base_out)

        for h in range(self.n_heads):
            A_h = self.lora_A[h * hr:(h + 1) * hr, :]      # [head_r, in]
            B_h = self.lora_B[:, h * hr:(h + 1) * hr]       # [out, head_r]
            G_h = B_h @ A_h                                   # [out, in], rank head_r
            alpha_h = G_h.norm()
            M_h = newtonschulz5(G_h, steps=self.ns_steps)
            lora_out = lora_out + alpha_h * F.linear(x_drop, M_h)

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.base_linear.in_features}, out={self.base_linear.out_features}, "
                f"r={self.r}, n_heads={self.n_heads}, ns_steps={self.ns_steps}")


# ============================================================
# Factored Orthogonal LoRA Linear Layer
# ============================================================

class FactoredLoRALinear(nn.Module):
    """
    Factored Orthogonal LoRA: NS on A and B separately + per-direction scaling.

    ΔW = B_orth @ diag(s) @ A_orth  — learned SVD with orthogonal bases.
    - A_orth = NS(A): orthonormal rows (input space basis)
    - B_orth = NS(B): orthonormal columns (output space basis)
    - s: learnable per-direction scale [r]

    Advantages over MuonLoRALinear:
    - Per-direction scaling (not uniform alpha)
    - NS on [r, n] matrices instead of [out, in] — much cheaper
    - No rank collapse (orthogonal bases) + full per-direction expressiveness
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_dropout: float = 0.05,
        ns_steps: int = 5,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.base_linear = base_linear
        base_linear.weight.requires_grad_(False)
        if base_linear.bias is not None:
            base_linear.bias.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.ns_steps = ns_steps

        # A [r, in]: orthogonal init, NS will keep rows orthonormal
        # B [out, r]: orthogonal init, NS will keep columns orthonormal
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        nn.init.orthogonal_(self.lora_A)
        nn.init.orthogonal_(self.lora_B)

        # Per-direction scale: init small for near-identity start
        self.lora_s = nn.Parameter(torch.full((r,), init_scale))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        # Orthogonalize A and B separately (cheap: iterate on [r,r] matrices)
        A_orth = newtonschulz5(self.lora_A, steps=self.ns_steps)  # [r, in]
        B_orth = newtonschulz5(self.lora_B, steps=self.ns_steps)  # [out, r]

        # Factored forward: x → project to r-dim → scale → project to out-dim
        # Equivalent to ΔW = B_orth @ diag(s) @ A_orth
        h = F.linear(self.lora_dropout(x), A_orth)  # [batch, r]
        h = h * self.lora_s                          # per-direction scale
        lora_out = F.linear(h, B_orth)               # [batch, out]

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (f"in={self.base_linear.in_features}, out={self.base_linear.out_features}, "
                f"r={self.r}, ns_steps={self.ns_steps}")


# ============================================================
# PatchSmoother (reuse from v3)
# ============================================================

class PatchSmoother(nn.Module):
    """Cross-patch conv to fix grid artifacts. Zero-init → identity start."""

    def __init__(self, channels: int = 18, hidden_dim: int = 64,
                 kernel_size: int = 7, num_layers: int = 3):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_ch = channels if i == 0 else hidden_dim
            out_ch = channels if i == num_layers - 1 else hidden_dim
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad,
                                    padding_mode='replicate'))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_2d = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x_2d = x_2d + self.net(x_2d)
        return x_2d.permute(0, 2, 3, 1).reshape(B, T, H, W, C)


# ============================================================
# Model Wrapper
# ============================================================

class PDELoRAModelMuon(nn.Module):
    """
    Muon-LoRA wrapper for PDEModelV3.

    Same interface as PDELoRAModelV3 — drop-in replacement.
    Replaces target Linear modules with MuonLoRALinear (NS orthogonalization).
    """

    def __init__(
        self,
        config: Dict,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        self.model = PDEModelV3(config)

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        self._apply_muon_lora(config)

        if freeze_encoder:
            self._freeze_encoders()
        if freeze_decoder:
            self._freeze_decoders()

        # Optional patch smoother
        smoother_cfg = config.get('model', {}).get('patch_smoother', {})
        if smoother_cfg.get('enabled', False):
            in_ch = config.get('model', {}).get('in_channels', 18)
            self.patch_smoother = PatchSmoother(
                channels=in_ch,
                hidden_dim=smoother_cfg.get('hidden_dim', 64),
                kernel_size=smoother_cfg.get('kernel_size', 7),
                num_layers=smoother_cfg.get('num_layers', 3),
            )
            if _is_main_process():
                n_ps = sum(p.numel() for p in self.patch_smoother.parameters())
                print(f"\nPatchSmoother enabled: {n_ps:,} params")

        if _is_main_process():
            self._log_param_summary()

    # ----------------------------------------------------------
    # Pretrained weight loading
    # ----------------------------------------------------------

    def _load_pretrained(self, pretrained_path: str):
        if _is_main_process():
            print(f"\nLoading pretrained weights from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned = {}
        for k, v in state_dict.items():
            k = k.removeprefix('module.').removeprefix('_orig_mod.')
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        if _is_main_process():
            if missing:
                print(f"  Missing keys (randomly init): {len(missing)}")
                for k in missing[:10]:
                    print(f"    - {k}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {len(unexpected)}")
                for k in unexpected[:10]:
                    print(f"    - {k}")
            loaded = len(cleaned) - len(unexpected)
            print(f"  Loaded {loaded} keys successfully")

    # ----------------------------------------------------------
    # Muon-LoRA injection (replaces PEFT)
    # ----------------------------------------------------------

    def _apply_muon_lora(self, config: Dict):
        """Replace target Linear modules with Muon-LoRA or Factored Orthogonal LoRA."""
        lora_cfg = config.get('model', {}).get('lora', {})
        r = lora_cfg.get('r', 16)
        dropout = lora_cfg.get('dropout', 0.05)
        ns_steps = lora_cfg.get('ns_steps', 5)
        init_scale = lora_cfg.get('init_scale', 0.01)
        lora_type = lora_cfg.get('type', 'muon')  # 'muon', 'factored', 'multihead', 'splitns'
        n_heads = lora_cfg.get('n_heads', 4)
        target_modules = set(lora_cfg.get('target_modules', [
            "qkv", "proj", "gate_proj", "up_proj", "down_proj",
        ]))

        type_map = {'muon': MuonLoRALinear, 'factored': FactoredLoRALinear,
                     'multihead': MultiHeadMuonLoRALinear, 'splitns': SplitNSLoRALinear}
        LayerClass = type_map.get(lora_type, MuonLoRALinear)

        replaced = 0
        for name, module in list(self.model.transformer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            short_name = name.split('.')[-1]
            if short_name not in target_modules:
                continue

            parts = name.split('.')
            parent = self.model.transformer
            for p in parts[:-1]:
                parent = getattr(parent, p)

            kwargs = dict(base_linear=module, r=r, lora_dropout=dropout,
                          ns_steps=ns_steps, init_scale=init_scale)
            if lora_type == 'multihead':
                kwargs['n_heads'] = n_heads
            if lora_cfg.get('use_gate', False) and lora_type == 'muon':
                kwargs['use_gate'] = True
            lora_linear = LayerClass(**kwargs)
            setattr(parent, parts[-1], lora_linear)
            replaced += 1

        # Freeze all non-LoRA transformer params
        for name, param in self.model.transformer.named_parameters():
            if 'lora_' not in name:
                param.requires_grad_(False)

        if _is_main_process():
            print(f"\n{lora_type.upper()}-LoRA config:")
            print(f"  type={lora_type}, r={r}, ns_steps={ns_steps}, init_scale={init_scale}")
            print(f"  target_modules={target_modules}")
            print(f"  Replaced {replaced} Linear layers")
            n_lora = sum(p.numel() for p in self.model.transformer.parameters()
                         if p.requires_grad)
            print(f"  LoRA params: {n_lora:,}")

    # ----------------------------------------------------------
    # Freeze helpers
    # ----------------------------------------------------------

    def _freeze_module(self, module: nn.Module, name: str):
        count = 0
        for param in module.parameters():
            param.requires_grad = False
            count += param.numel()
        if _is_main_process():
            print(f"  Froze {name}: {count:,} parameters")

    def _freeze_encoders(self):
        for attr in ['encoder', 'encoder_1d', 'encoder_3d']:
            if hasattr(self.model, attr):
                self._freeze_module(getattr(self.model, attr),
                                    f"Encoder ({attr.split('_')[-1].upper() if '_' in attr else '2D'})")

    def _freeze_decoders(self):
        for attr in ['decoder', 'decoder_1d', 'decoder_3d']:
            if hasattr(self.model, attr):
                self._freeze_module(getattr(self.model, attr),
                                    f"Decoder ({attr.split('_')[-1].upper() if '_' in attr else '2D'})")

    # ----------------------------------------------------------
    # Param summary
    # ----------------------------------------------------------

    def _log_param_summary(self):
        components = {}
        if hasattr(self.model, 'encoder'):
            components['Encoder (2D)'] = self.model.encoder
        if hasattr(self.model, 'decoder'):
            components['Decoder (2D)'] = self.model.decoder
        for dim_name in ['1d', '3d']:
            for comp_type, label in [('encoder', 'Encoder'), ('decoder', 'Decoder')]:
                attr = f'{comp_type}_{dim_name}'
                if hasattr(self.model, attr):
                    components[f'{label} ({dim_name.upper()})'] = getattr(self.model, attr)
        components['Transformer'] = self.model.transformer

        total_all, trainable_all = 0, 0
        print(f"\n{'='*60}")
        print(f"PDELoRAModelMuon Parameter Summary")
        print(f"{'='*60}")

        for name, module in components.items():
            t = sum(p.numel() for p in module.parameters())
            tr = sum(p.numel() for p in module.parameters() if p.requires_grad)
            suffix = " (Muon-LoRA)" if name == 'Transformer' else ""
            print(f"{name + suffix + ':':<25} {t:>12,} total, {tr:>10,} trainable")
            total_all += t
            trainable_all += tr

        if hasattr(self, 'patch_smoother'):
            t = sum(p.numel() for p in self.patch_smoother.parameters())
            tr = sum(p.numel() for p in self.patch_smoother.parameters() if p.requires_grad)
            print(f"{'PatchSmoother:':<25} {t:>12,} total, {tr:>10,} trainable")
            total_all += t
            trainable_all += tr

        print(f"{'-'*60}")
        print(f"{'Total:':<25} {total_all:>12,} total, {trainable_all:>10,} trainable")
        print(f"Trainable ratio: {100 * trainable_all / total_all:.2f}%")
        print(f"{'='*60}\n")

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_normalized: bool = False,
    ) -> torch.Tensor:
        result = self.model(x, channel_mask, return_normalized)

        if not hasattr(self, 'patch_smoother'):
            return result

        if return_normalized:
            output_norm, mean, std = result
            if output_norm.ndim == 5:
                output_norm = self.patch_smoother(output_norm)
            return output_norm, mean, std
        else:
            if result.ndim == 5:
                result = self.patch_smoother(result)
            return result

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Checkpoint save / load
# ============================================================

def save_lora_checkpoint(
    model: PDELoRAModelMuon,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    metrics: Dict,
    save_path: str,
    config: Dict,
    patience_counter: int = 0,
    best_val_loss: float = float('inf'),
):
    """Save Muon-LoRA checkpoint (trainable params only)."""
    trainable_state_dict = {}
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.data.clone()

    checkpoint = {
        'global_step': global_step,
        'trainable_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'patience_counter': patience_counter,
        'best_val_loss': best_val_loss,
        'model_type': 'muon_lora',
    }

    if hasattr(model, 'patch_smoother'):
        checkpoint['patch_smoother_state_dict'] = model.patch_smoother.state_dict()

    torch.save(checkpoint, save_path)


def load_lora_checkpoint(
    model: PDELoRAModelMuon,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    """Load Muon-LoRA checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'trainable_state_dict' in checkpoint:
        trainable_state_dict = checkpoint['trainable_state_dict']
        model_state = model.model.state_dict()

        matched = {k: v for k, v in trainable_state_dict.items() if k in model_state}
        skipped = set(trainable_state_dict.keys()) - set(matched.keys())

        model_state.update(matched)
        model.model.load_state_dict(model_state)

        if _is_main_process():
            print(f"Loaded {len(matched)}/{len(trainable_state_dict)} trainable parameters")
            if skipped:
                print(f"  Skipped {len(skipped)} keys:")
                for k in sorted(skipped)[:10]:
                    print(f"    - {k}")

    if hasattr(model, 'patch_smoother') and 'patch_smoother_state_dict' in checkpoint:
        try:
            model.patch_smoother.load_state_dict(checkpoint['patch_smoother_state_dict'])
            if _is_main_process():
                print("Loaded patch smoother state dict")
        except RuntimeError as e:
            if _is_main_process():
                print(f"Warning: Smoother shape mismatch: {e}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load optimizer state: {e}")

    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            if _is_main_process():
                print(f"Warning: Failed to load scheduler state: {e}")

    return checkpoint
