"""
OrbitalGuard AI — TLE Preprocessing Pipeline
=============================================
Converts raw TLE data into normalized feature sequences for LSTM training.

Pipeline:
  TLE file → SGP4 propagation (N timesteps) → feature extraction
  → MinMax normalization → sliding-window sequences → .npy output

Features extracted per timestep:
  [x, y, z, vx, vy, vz]  (ECI position km, velocity km/s)

Sequence shape: (N_samples, seq_len=10, 6)
Target shape:   (N_samples, 3)  — residual [dx, dy, dz] at T+horizon
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta

# ─── Path Alignment ───────────────────────────────────────
PREPROC_DIR  = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT    = os.path.dirname(PREPROC_DIR)
PROJECT_ROOT = os.path.dirname(AIML_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, 'backend')

if BACKEND_ROOT not in sys.path: sys.path.insert(0, BACKEND_ROOT)

from sgp4.api import Satrec, jday

# ─── Constants ────────────────────────────────────────────
TLE_PATH     = os.path.join(BACKEND_ROOT, 'data', 'tle_data.txt')
OUTPUT_DIR   = os.path.join(AIML_ROOT, 'data')
SEQ_LEN      = 10          # timesteps fed to LSTM
HORIZON      = 1           # prediction horizon (steps ahead)
DT_SECONDS   = 60          # 1 minute per step
MAX_OBJECTS  = 3000        # objects to sample from TLE file
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO = 0.15 (remainder)


# ──────────────────────────────────────────────────────────
def jday_from_dt(dt: datetime):
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


def load_tle_entries(path: str, max_objects: int = MAX_OBJECTS) -> list:
    """Parse TLE file (with optional blank lines) into list of {name, line1, line2}."""
    with open(path, 'r', errors='ignore') as f:
        raw = [l.strip() for l in f.readlines()]

    entries, i = [], 0
    while i < len(raw) and len(entries) < max_objects:
        # skip blank lines
        if not raw[i]:
            i += 1
            continue
        name = raw[i]
        # advance to next non-blank
        i += 1
        while i < len(raw) and not raw[i]: i += 1
        if i >= len(raw): break
        line1 = raw[i]; i += 1
        while i < len(raw) and not raw[i]: i += 1
        if i >= len(raw): break
        line2 = raw[i]; i += 1

        if line1.startswith('1') and line2.startswith('2'):
            entries.append({'name': name, 'line1': line1, 'line2': line2})

    return entries


def propagate_sequence(sat: Satrec, start_dt: datetime,
                        n_steps: int, dt_sec: float) -> np.ndarray:
    """
    Propagate satellite for n_steps at dt_sec intervals.
    Returns array of shape (n_steps, 6): [x,y,z,vx,vy,vz]
    Returns None if any propagation fails.
    """
    states = []
    for i in range(n_steps):
        t = start_dt + timedelta(seconds=i * dt_sec)
        jd, fr = jday_from_dt(t)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0 or any(np.isnan(r)) or any(np.isnan(v)):
            return None
        states.append(list(r) + list(v))
    return np.array(states, dtype=np.float32)


def build_sequences(states: np.ndarray, seq_len: int, horizon: int):
    """
    Build input sequences and residual targets from a propagated state array.

    The 'residual' here is simulated: we add a small synthetic drift to
    the SGP4 positions to represent atmospheric drag / radiation pressure,
    and the target is the magnitude of that accumulated drift at T+horizon.

    In production this would be replaced by actual radar vs. SGP4 deltas.
    """
    X, y = [], []
    n = len(states)
    needed = seq_len + horizon

    if n < needed:
        return None, None

    # Simulate drift (same logic as lstm_model.py for consistency)
    decay  = np.random.uniform(0.005, 0.015)
    t_arr  = np.arange(n, dtype=np.float32)
    drift  = np.zeros((n, 3), dtype=np.float32)
    drift[:, 0] = -decay * (t_arr ** 2) + np.random.uniform(0.5, 1.5) * np.sin(t_arr * 2)
    drift[:, 1] = -decay * (t_arr ** 2.1) + np.random.uniform(0.5, 1.5) * np.cos(t_arr * 2)
    drift[:, 2] = -(decay / 2) * t_arr + 0.1 * np.sin(t_arr)
    drift += np.random.normal(0, 0.05, drift.shape).astype(np.float32)

    # Perturbed (true) states
    true_states = states.copy()
    true_states[:, :3] += drift

    for start in range(0, n - needed + 1, 2):  # stride=2 for diversity
        seq    = true_states[start : start + seq_len]          # (seq_len, 6)
        target = drift[start + seq_len + horizon - 1]          # (3,) residual at T+horizon
        X.append(seq)
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ──────────────────────────────────────────────────────────
class TLEPreprocessor:
    def __init__(self,
                 tle_path: str = TLE_PATH,
                 output_dir: str = OUTPUT_DIR,
                 seq_len: int = SEQ_LEN,
                 horizon: int = HORIZON,
                 dt_sec: float = DT_SECONDS,
                 max_objects: int = MAX_OBJECTS,
                 n_propagation_steps: int = 60):
        self.tle_path   = tle_path
        self.output_dir = output_dir
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.dt_sec     = dt_sec
        self.max_objects = max_objects
        self.n_steps    = n_propagation_steps  # steps per satellite
        self.start_dt   = datetime.utcnow()

        # Normalization parameters (fitted on training set)
        self.x_mean: np.ndarray = None
        self.x_std:  np.ndarray = None

    # ── Core Pipeline ─────────────────────────────────────
    def run(self) -> dict:
        """End-to-end pipeline. Returns paths to saved .npy files."""
        print("=" * 60)
        print("[TLE Preprocessor] Starting pipeline...")
        print(f"  TLE source  : {self.tle_path}")
        print(f"  Max objects : {self.max_objects}")
        print(f"  Seq length  : {self.seq_len}  |  Horizon: {self.horizon}")
        print(f"  Steps/sat   : {self.n_steps}  |  dt: {self.dt_sec}s")
        print("=" * 60)

        # 1. Load TLE
        entries = load_tle_entries(self.tle_path, self.max_objects)
        print(f"[1/5] Loaded {len(entries)} TLE entries")

        # 2. Propagate + build sequences
        all_X, all_y = [], []
        skipped = 0

        for entry in entries:
            try:
                sat = Satrec.twoline2rv(entry['line1'], entry['line2'])
            except Exception:
                skipped += 1
                continue

            states = propagate_sequence(sat, self.start_dt,
                                        self.n_steps + self.horizon,
                                        self.dt_sec)
            if states is None:
                skipped += 1
                continue

            X, y = build_sequences(states, self.seq_len, self.horizon)
            if X is None or len(X) == 0:
                skipped += 1
                continue

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise RuntimeError("[Preprocessor] No valid sequences generated. Check TLE file.")

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        print(f"[2/5] Propagated | Valid: {len(entries)-skipped} | Skipped: {skipped}")
        print(f"      Raw sequences: X={X.shape}  y={y.shape}")

        # 3. Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        # 4. Normalize (fit on train split only)
        n_train = int(len(X) * TRAIN_RATIO)
        n_val   = int(len(X) * VAL_RATIO)

        X_train = X[:n_train]
        X_val   = X[n_train : n_train + n_val]
        X_test  = X[n_train + n_val:]
        y_train = y[:n_train]
        y_val   = y[n_train : n_train + n_val]
        y_test  = y[n_train + n_val:]

        # Fit normalizer on train set (per feature, across all timesteps)
        flat_train = X_train.reshape(-1, X_train.shape[-1])
        self.x_mean = flat_train.mean(axis=0)
        self.x_std  = flat_train.std(axis=0) + 1e-8

        X_train = self._normalize(X_train)
        X_val   = self._normalize(X_val)
        X_test  = self._normalize(X_test)

        print(f"[3/5] Normalized  | Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

        # 5. Save
        os.makedirs(self.output_dir, exist_ok=True)
        paths = {
            'X_train': os.path.join(self.output_dir, 'sequence_X_train.npy'),
            'X_val':   os.path.join(self.output_dir, 'sequence_X_val.npy'),
            'X_test':  os.path.join(self.output_dir, 'sequence_X_test.npy'),
            'y_train': os.path.join(self.output_dir, 'sequence_y_train.npy'),
            'y_val':   os.path.join(self.output_dir, 'sequence_y_val.npy'),
            'y_test':  os.path.join(self.output_dir, 'sequence_y_test.npy'),
            'norm':    os.path.join(self.output_dir, 'normalizer.npz'),
        }

        np.save(paths['X_train'], X_train)
        np.save(paths['X_val'],   X_val)
        np.save(paths['X_test'],  X_test)
        np.save(paths['y_train'], y_train)
        np.save(paths['y_val'],   y_val)
        np.save(paths['y_test'],  y_test)
        np.savez(paths['norm'], mean=self.x_mean, std=self.x_std)

        print(f"[4/5] Saved all arrays to {self.output_dir}")
        print(f"[5/5] ✅ Preprocessing complete.")
        print("=" * 60)
        return paths

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize using training statistics."""
        return (X - self.x_mean) / self.x_std

    def load_normalizer(self, path: str = None):
        """Load previously saved normalization parameters."""
        path = path or os.path.join(self.output_dir, 'normalizer.npz')
        data = np.load(path)
        self.x_mean = data['mean']
        self.x_std  = data['std']
        return self


# ──────────────────────────────────────────────────────────
def load_splits(data_dir: str = OUTPUT_DIR):
    """Convenience loader: returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
    keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
    arrays = []
    for k in keys:
        path = os.path.join(data_dir, f'sequence_{k}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Preprocessor] Missing {path}. Run preprocessing first.")
        arrays.append(np.load(path))
    return tuple(arrays)


# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    preprocessor = TLEPreprocessor(
        max_objects=2000,
        n_propagation_steps=50
    )
    paths = preprocessor.run()
    for k, v in paths.items():
        print(f"  {k}: {v}")
