from __future__ import annotations

import math

import torch
import torch.nn as nn


class GraphAttentionBlock(nn.Module):
    """Compact graph attention block over spectro-temporal tokens."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens)
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.channels)
        out = self.out_proj(context)
        return tokens + out


class AASISTModel(nn.Module):
    """Waveform-first AASIST-style model for ASVspoof 2019 LA.

    The reference AASIST architecture combines a raw-audio front-end and a spectro-temporal
    graph attention network. This implementation preserves that conceptual structure while
    remaining computationally manageable in the current research environment.
    """

    def __init__(self, input_channels: int = 1, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.waveform_frontend = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.spectral_projection = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.node_mixer = nn.Linear(96, hidden_dim)
        self.graph_attention = nn.Sequential(
            GraphAttentionBlock(hidden_dim, num_heads=num_heads, dropout=0.1),
            GraphAttentionBlock(hidden_dim, num_heads=num_heads, dropout=0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _spectro_temporal_tokens(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(256, device=x.device)
        spec = torch.stft(
            x.squeeze(1),
            n_fft=256,
            hop_length=64,
            win_length=256,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = torch.abs(spec).unsqueeze(1)
        mag = self.spectral_projection(mag)
        mag = torch.nn.functional.adaptive_avg_pool2d(mag, (32, 32))
        b, c, f, t = mag.shape
        tokens = mag.permute(0, 2, 3, 1).reshape(b, f * t, c)
        tokens = torch.nn.functional.adaptive_avg_pool1d(tokens.permute(0, 2, 1), 64).permute(0, 2, 1)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected waveform input shape (batch, samples); got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        waveform_features = self.waveform_frontend(x)
        waveform_features = torch.nn.functional.adaptive_avg_pool1d(waveform_features, 64)
        wave_tokens = waveform_features.permute(0, 2, 1).contiguous()  # [B, 64, 64]
        spec_tokens = self._spectro_temporal_tokens(x)  # [B, 64, 32]
        graph_tokens = torch.cat([wave_tokens, spec_tokens], dim=-1)  # [B, 64, 96]
        graph_tokens = self.node_mixer(graph_tokens)
        graph_tokens = self.graph_attention(graph_tokens)
        pooled = torch.cat([graph_tokens.mean(dim=1), graph_tokens.std(dim=1)], dim=-1)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)


__all__ = ["AASISTModel", "GraphAttentionBlock"]
