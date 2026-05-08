from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = PROJECT_ROOT / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from neural_cnn import DEFAULT_BOOTSTRAP, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, NeuralConfig, run_neural_baseline  # noqa: E402


class NeuralTransformer(nn.Module):
    def __init__(self, input_dim: int = 8, model_dim: int = 64, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(torch.zeros(1, 56, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        tokens = self.input_projection(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0, :]).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the supervised Transformer sequence baseline.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    args = parser.parse_args()
    config = NeuralConfig(
        baseline_name="neural_transformer",
        checkpoint_path=PROJECT_ROOT / "models" / "neural_transformer_v1.pt",
        result_path=PROJECT_ROOT / "results" / "neural_transformer_results.json",
        seed=22902692,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_bootstrap=args.bootstrap,
    )
    result = run_neural_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
