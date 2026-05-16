"""Train LSTM and Transformer on sequences of last 10 games per team.

Inputs: data/sequences.npz
Outputs:
  models/lstm.pt, models/transformer.pt
  results/deep_metrics.json
  results/deep_preds_test.csv
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"


class LSTMModel(nn.Module):
    """Process each team's history with the same LSTM, then combine."""
    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, 20, F) = home_seq(10) || away_seq(10)
        home = x[:, :10, :]
        away = x[:, 10:, :]
        _, (h_home, _) = self.lstm(home)
        _, (h_away, _) = self.lstm(away)
        # take last layer hidden
        h_home = h_home[-1]
        h_away = h_away[-1]
        z = torch.cat([h_home, h_away], dim=-1)
        return self.head(z).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 64, nhead: int = 4, nlayers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos = PositionalEncoding(d_model, max_len=32)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # process home and away separately with the same encoder
        home = self.encoder(self.pos(self.input_proj(x[:, :10, :])))
        away = self.encoder(self.pos(self.input_proj(x[:, 10:, :])))
        z = torch.cat([home.mean(dim=1), away.mean(dim=1)], dim=-1)
        return self.head(z).squeeze(-1)


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    # flatten over time, fit on train only
    flat_tr = X_train.reshape(-1, X_train.shape[-1])
    mu = flat_tr.mean(axis=0)
    sd = flat_tr.std(axis=0)
    sd[sd == 0] = 1.0
    Xn_tr = (X_train - mu) / sd
    Xn_te = (X_test - mu) / sd
    return Xn_tr.astype(np.float32), Xn_te.astype(np.float32), mu, sd


def fit_model(model: nn.Module, X_tr, y_tr, X_va, y_va, epochs: int = 40, lr: float = 1e-3,
              batch_size: int = 64, patience: int = 6, weight_decay: float = 1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)
        sched.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(torch.from_numpy(X_va))
            v_loss = float(nn.BCEWithLogitsLoss()(v_logits, torch.from_numpy(y_va)).item())
        if ep % 5 == 0 or v_loss < best_val:
            print(f"    ep{ep:02d} train {ep_loss:.4f} val {v_loss:.4f}")
        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop @ epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate(model: nn.Module, X, y) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).cpu().numpy()
    p = 1 / (1 + np.exp(-logits))
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }, p


def main():
    z = np.load(DATA / "sequences.npz")
    X_tr_all, y_tr_all = z["X_train"], z["y_train"]
    X_te, y_te = z["X_test"], z["y_test"]
    print(f"sequences train {X_tr_all.shape} test {X_te.shape}")

    # carve out validation (last 15% chronologically)
    n_val = int(0.15 * len(X_tr_all))
    X_tr, X_va = X_tr_all[:-n_val], X_tr_all[-n_val:]
    y_tr, y_va = y_tr_all[:-n_val], y_tr_all[-n_val:]
    print(f"  fit on {len(X_tr)}, val {len(X_va)}, test {len(X_te)}")

    X_tr_n, X_va_n, mu, sd = normalize(X_tr, X_va)
    _, X_te_n, _, _ = normalize(X_tr, X_te)  # re-normalize test w/ train stats
    X_te_n = ((X_te - mu) / sd).astype(np.float32)

    n_feat = X_tr.shape[-1]
    metrics = {}
    preds = pd.DataFrame({"y_true": y_te})

    # ---------------- LSTM ----------------
    print("\n[LSTM]")
    t0 = time.time()
    lstm = LSTMModel(n_feat=n_feat, hidden=64, dropout=0.3)
    lstm, best_val = fit_model(lstm, X_tr_n, y_tr.astype(np.float32),
                                X_va_n, y_va.astype(np.float32),
                                epochs=40, lr=1e-3, batch_size=64)
    m, p = evaluate(lstm, X_te_n, y_te)
    m["best_val_loss"] = best_val
    m["fit_seconds"] = time.time() - t0
    metrics["lstm"] = m
    preds["lstm_p"] = p
    torch.save(lstm.state_dict(), MODELS / "lstm.pt")
    print(f"[lstm] {m}")

    # ---------------- Transformer ----------------
    print("\n[Transformer]")
    t0 = time.time()
    trf = TransformerModel(n_feat=n_feat, d_model=64, nhead=4, nlayers=2, dropout=0.3)
    trf, best_val = fit_model(trf, X_tr_n, y_tr.astype(np.float32),
                               X_va_n, y_va.astype(np.float32),
                               epochs=40, lr=1e-3, batch_size=64)
    m, p = evaluate(trf, X_te_n, y_te)
    m["best_val_loss"] = best_val
    m["fit_seconds"] = time.time() - t0
    metrics["transformer"] = m
    preds["transformer_p"] = p
    torch.save(trf.state_dict(), MODELS / "transformer.pt")
    print(f"[transformer] {m}")

    preds.to_csv(RESULTS / "deep_preds_test.csv", index=False)
    with open(RESULTS / "deep_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # save normalization for downstream re-use
    np.savez(MODELS / "deep_normalize.npz", mu=mu, sd=sd)
    print("\nSaved deep model metrics + predictions.")


if __name__ == "__main__":
    main()
