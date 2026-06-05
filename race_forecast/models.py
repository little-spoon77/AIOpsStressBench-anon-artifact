from __future__ import annotations

import math

import torch
from torch import nn


class LastValue(nn.Module):
    def __init__(self, pred_len: int, target_metric: int = 0) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.target_metric = target_metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, self.target_metric]
        return last.repeat(1, self.pred_len)


class SeasonalNaive(nn.Module):
    def __init__(self, pred_len: int, target_metric: int = 0, season_len: int = 24) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.target_metric = target_metric
        self.season_len = season_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :, self.target_metric]
        if target.shape[1] < self.season_len:
            return target[:, -1:].repeat(1, self.pred_len)
        pattern = target[:, -self.season_len :]
        repeats = math.ceil(self.pred_len / self.season_len)
        return pattern.repeat(1, repeats)[:, : self.pred_len]


class DLinear(nn.Module):
    def __init__(self, input_len: int, pred_len: int, n_metrics: int) -> None:
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_metrics = n_metrics
        self.linear = nn.Linear(input_len, pred_len)
        self.proj = nn.Linear(n_metrics, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(temporal).squeeze(-1)


class RaceDLinear(nn.Module):
    def __init__(self, input_len: int, pred_len: int, n_metrics: int, use_mask: bool = True) -> None:
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_metrics = n_metrics
        self.base_linear = nn.Linear(input_len, pred_len)
        self.base_proj = nn.Linear(n_metrics, 1)
        self.use_mask = use_mask
        self.mask_linear = nn.Linear(input_len, pred_len)
        self.mask_proj = nn.Linear(n_metrics, 1)
        nn.init.zeros_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reliability = (x.abs() > 1e-6).float()
        missing = 1.0 - reliability
        base = self.base_linear(x.transpose(1, 2)).transpose(1, 2)
        base = self.base_proj(base).squeeze(-1)
        if not self.use_mask:
            return base
        correction = self.mask_linear(missing.transpose(1, 2)).transpose(1, 2)
        correction = self.mask_proj(correction).squeeze(-1)
        missing_ratio = missing.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return base + correction * missing_ratio


class RaceNLinear(nn.Module):
    def __init__(self, input_len: int, pred_len: int, n_metrics: int, target_metric: int = 0) -> None:
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_metrics = n_metrics
        self.target_metric = target_metric
        self.residual_linear = nn.Linear(input_len, pred_len)
        self.residual_proj = nn.Linear(n_metrics, 1)
        self.mask_linear = nn.Linear(input_len, pred_len)
        self.mask_proj = nn.Linear(n_metrics, 1)
        nn.init.zeros_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reliability = (x.abs() > 1e-6).float()
        last = x[:, -1:, :]
        centered = x - last
        residual = self.residual_linear(centered.transpose(1, 2)).transpose(1, 2)
        residual = self.residual_proj(residual).squeeze(-1)
        missing = 1.0 - reliability
        correction = self.mask_linear(missing.transpose(1, 2)).transpose(1, 2)
        correction = self.mask_proj(correction).squeeze(-1)
        missing_ratio = missing.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        anchor = x[:, -1:, self.target_metric].repeat(1, self.pred_len)
        return anchor + residual + correction * missing_ratio


class TinyPatchTST(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_metrics: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_metrics = n_metrics
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = 1 + max(0, (input_len - patch_len) // stride)
        patch_dim = patch_len * n_metrics
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * self.n_patches),
            nn.Linear(d_model * self.n_patches, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(x.shape[0], self.n_patches, self.patch_len * self.n_metrics)
        tokens = self.patch_proj(patches) + self.pos
        encoded = self.encoder(tokens)
        return self.head(encoded.reshape(x.shape[0], -1))


def build_model(name: str, input_len: int, pred_len: int, n_metrics: int, target_metric: int) -> nn.Module:
    lowered = name.lower()
    if lowered == "last_value":
        return LastValue(pred_len=pred_len, target_metric=target_metric)
    if lowered == "seasonal_naive":
        return SeasonalNaive(pred_len=pred_len, target_metric=target_metric, season_len=min(288, input_len))
    if lowered == "dlinear":
        return DLinear(input_len=input_len, pred_len=pred_len, n_metrics=n_metrics)
    if lowered == "race_dlinear":
        return RaceDLinear(input_len=input_len, pred_len=pred_len, n_metrics=n_metrics)
    if lowered == "race_dlinear_nomask":
        return RaceDLinear(input_len=input_len, pred_len=pred_len, n_metrics=n_metrics, use_mask=False)
    if lowered == "race_nlinear":
        return RaceNLinear(input_len=input_len, pred_len=pred_len, n_metrics=n_metrics, target_metric=target_metric)
    if lowered == "patchtst":
        return TinyPatchTST(input_len=input_len, pred_len=pred_len, n_metrics=n_metrics)
    raise ValueError(f"Unsupported model: {name}")


def is_trainable(model: nn.Module) -> bool:
    return any(param.requires_grad for param in model.parameters())
