"""
model.py
BiLSTM architecture — must match training exactly. Do NOT change.
"""
import torch
import torch.nn as nn

LOOKBACK = 24   # 24 × 5 min = 2 hours of history
HORIZON  = 12   # 12 × 5 min = 1 hour ahead
N_FEAT   = 17   # 17 input features
DROP     = 0.2

SCALABLE_FEATURES = [
    "GHI (W.m-2)", "GHI (W.m-2)_lag1", "GHI (W.m-2)_lag2", "GHI (W.m-2)_roll6",
    "DHI_SPN1 (W.m-2)", "DHI_SPN1 (W.m-2)_lag1",
    "PAR (umol.s-1.m-2)_lag1", "PAR (umol.s-1.m-2)_roll6",
    "Albedometer (W.m-2)", "airtemp_underpanel",
    "sin_hour", "cos_hour",
    "GHI_diff1", "GHI_diff2", "PAR_diff1", "cloud_flag",
]
# treatment_enc is feature 17 — appended last, NOT scaled (0 = Fixed-tilt, 1 = Vertical)


class DualHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared   = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(DROP),
            nn.Linear(64, 32),     nn.ReLU(),
        )
        self.pv_head  = nn.Sequential(nn.Linear(32, HORIZON), nn.ReLU())
        self.par_head = nn.Sequential(nn.Linear(32, HORIZON), nn.ReLU())

    def forward(self, x):
        s = self.shared(x)
        return self.pv_head(s), self.par_head(s)


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM — best model from 5-architecture comparison.
    PV power R² = 0.9085  |  PAR R² = 0.8926
    Reads the 2-hour sensor window forward AND backward simultaneously.
    """
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(N_FEAT, 128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(DROP)
        self.lstm2 = nn.LSTM(256, 64,  batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(DROP)
        self.head  = DualHead(128)  # 128 = 64 * 2 (bidirectional)

    def forward(self, x):
        o, _ = self.lstm1(x);  o = self.drop1(o)
        o, _ = self.lstm2(o);  o = self.drop2(o[:, -1])
        return self.head(o)    # returns (pv, par) each shape (batch, 12)
