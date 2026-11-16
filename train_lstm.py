"""LSTM voltage forecasting prototype: 5 past steps -> next step, compared with persistence.
With ONE deterministic day the numbers are NOT meaningful. Generate trajectories first:
    python ev_load_generator.py --n-traj 100"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SINGLE, MULTI = "ieee33_ev_voltage_dataset.npy", "ieee33_ev_voltage_dataset_multi.npy"
SEED, SEQ_LEN, EPOCHS = 42, 5, 100


def make_windows(traj, seq_len):
    """traj: (T, n_buses) -> X (T-seq_len, seq_len, n_buses), y (T-seq_len, n_buses)."""
    n = len(traj) - seq_len
    return (np.stack([traj[i:i + seq_len] for i in range(n)]),
            np.stack([traj[i + seq_len] for i in range(n)]))


def windows_from(trajs, seq_len):
    pairs = [make_windows(tr, seq_len) for tr in trajs]  # windows never span two trajectories
    return (torch.tensor(np.concatenate([p[0] for p in pairs]), dtype=torch.float32),
            torch.tensor(np.concatenate([p[1] for p in pairs]), dtype=torch.float32))


def load_split(train_frac=0.8):
    if os.path.exists(MULTI):
        data = np.load(MULTI)                      # (n_traj, T, n_buses), split BY trajectory
        cut = int(train_frac * len(data))
        return list(data[:cut]), list(data[cut:])
    print("WARNING: only one trajectory found; using a time split. Results are not meaningful.")
    single = np.load(SINGLE)
    cut = int(train_frac * len(single))
    return [single[:cut]], [single[cut:]]


class VoltageLSTM(nn.Module):
    def __init__(self, n_features=33, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return x[:, -1, :] + self.fc(out[:, -1, :])   # predict the change from the last step


def rmse(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)))


if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_trajs, test_trajs = load_split()
    Xtr, ytr = windows_from(train_trajs, SEQ_LEN)
    Xte, yte = windows_from(test_trajs, SEQ_LEN)
    print(f"Train windows: {len(Xtr)}, test windows: {len(Xte)}")
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)

    model = VoltageLSTM(n_features=Xtr.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"epoch {epoch:3d}  train MSE {total / len(Xtr):.3e}  test RMSE {rmse(model(Xte), yte):.5f} p.u.")

    model.eval()
    with torch.no_grad():
        print(f"Final test RMSE - LSTM: {rmse(model(Xte), yte):.5f} p.u. | "
              f"persistence baseline: {rmse(Xte[:, -1, :], yte):.5f} p.u.")
    torch.save(model.state_dict(), "voltage_lstm.pt")
