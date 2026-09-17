"""
Train the Hermite reconstruction safeguard (MLP corrector) from a
PREBUILT dataset (see data_processing.py / run_data_processing.sh).

This script no longer builds the dataset itself -- run
`bash hermite_ml/run_data_processing.sh` (or data_processing.py directly)
first to produce hermite_ml/datasets/<name>/{train,val,config}.npz, then
point DATASET_NAME at it here.

Validation: val.npz (if present) is the CROSS-TIMESTEP check -- entirely
different bulk files than train.npz (see EXPERIMENT_LOG.md: cell-level
splitting within one snapshot hid a real overfitting problem, this is why
data_processing.py defaults to a timestep-level split). Best-checkpoint
selection uses val.npz when available, falling back to dev.npz (a
same-snapshot dev split, if data_processing.py was run with
--val-fraction-within-train) only as a last resort.

Usage
-----
python3 hermite_ml/corrector_train.py
"""

import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
torch.set_num_threads(4)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hermite_ml.corrector_model import HermiteCorrectorMLP

# ===
# CONFIG
# ===
DATASET_NAME  = 'multi_snapshot_v1'   # under hermite_ml/datasets/
HIDDEN        = 64
EPOCHS        = 400                   # ALWAYS smoke-test on a short run first
                                       # (agreed convention) before bumping this up.
BATCH_SIZE    = 4096
LR            = 1e-3
MODEL_OUT     = os.path.join(os.path.dirname(__file__), 'corrector_weights.pt')


def load_split(dataset_dir, name):
    path = os.path.join(dataset_dir, f'{name}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return dict(X=d['X'], y=d['y'], lam=d['lam'])


def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), 'datasets', DATASET_NAME)
    if not os.path.isdir(dataset_dir):
        raise SystemExit(
            f"Dataset not found: {dataset_dir}\n"
            f"Run: bash hermite_ml/run_data_processing.sh  (edit its config first)")

    cfg = dict(np.load(os.path.join(dataset_dir, 'config.npz'), allow_pickle=True))
    print(f"Dataset: {DATASET_NAME}")
    print(f"  train_bulkfiles = {list(cfg['train_bulkfiles'])}")
    print(f"  val_bulkfiles   = {list(cfg['val_bulkfiles'])}")
    print(f"  s_low={int(cfg['s_low'])}  full_order={int(cfg['full_order'])}")

    train = load_split(dataset_dir, 'train')
    val   = load_split(dataset_dir, 'val')
    dev   = load_split(dataset_dir, 'dev')
    if train is None:
        raise SystemExit(f"No train.npz in {dataset_dir}")

    if val is not None:
        eval_split, eval_label = val, 'val (cross-timestep)'
    elif dev is not None:
        eval_split, eval_label = dev, 'dev (same-snapshot, WEAKER signal)'
        print("WARNING: no val.npz (cross-timestep) found -- falling back to "
              "dev.npz. Best-checkpoint selection will NOT catch cross-timestep "
              "overfitting the way val.npz would (see EXPERIMENT_LOG.md).")
    else:
        raise SystemExit(
            f"No val.npz or dev.npz in {dataset_dir} -- can't select a best "
            f"checkpoint. Re-run data_processing.py with --val-bulkfiles or "
            f"--val-fraction-within-train.")

    print(f"train samples: {train['X'].shape[0]}   "
          f"{eval_label} samples: {eval_split['X'].shape[0]}")
    print(f"lambda range: train [{train['lam'].min():.3f}, {train['lam'].max():.3f}]")

    n_features = train['X'].shape[1] - 3
    model = HermiteCorrectorMLP(n_features=n_features, hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    def total_loss(pred, target, lam):
        fit = torch.mean((pred - target) ** 2)
        reg = torch.mean(lam * pred ** 2)
        return fit + reg, fit, reg

    Xt = torch.from_numpy(train['X']); yt = torch.from_numpy(train['y']); lt = torch.from_numpy(train['lam'])
    Xv = torch.from_numpy(eval_split['X']); yv = torch.from_numpy(eval_split['y']); lv = torch.from_numpy(eval_split['lam'])

    n_train = Xt.shape[0]
    print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE} "
          f"(eval on {eval_label}) ...")
    best_val_fit = float('inf')
    best_epoch = -1
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb, lb = Xt[idx], yt[idx], lt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss, _, _ = total_loss(pred, yb, lb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            _, val_fit, val_reg = total_loss(val_pred, yv, lv)
            val_loss_plain = torch.mean((val_pred - yv) ** 2).item()

        val_fit_v = val_fit.item()
        marker = ''
        if val_fit_v < best_val_fit:
            best_val_fit = val_fit_v
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = '  * new best'

        print_every = max(1, EPOCHS // 20)
        if epoch % print_every == 0 or epoch == EPOCHS - 1 or marker:
            print(f"  epoch {epoch:3d}: train_loss={epoch_loss:.5f}  "
                  f"val_fit={val_fit_v:.5f}  val_reg={val_reg.item():.5f}  "
                  f"val_mse_plain={val_loss_plain:.5f}{marker}")

    print(f"\nBest val_fit={best_val_fit:.5f} at epoch {best_epoch} "
          f"(final was epoch {EPOCHS-1})")
    torch.save({'state_dict': best_state, 's_low': int(cfg['s_low']),
               'full_order': int(cfg['full_order']), 'n_features': n_features,
               'hidden': HIDDEN, 'dataset_name': DATASET_NAME}, MODEL_OUT)
    print(f"Saved BEST-val model -> {MODEL_OUT}")


if __name__ == '__main__':
    main()
