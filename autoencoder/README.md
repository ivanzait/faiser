# AE pilot: autoencoder vs Hermite at equal storage

Feasibility test on the dense 1D two-stream run (MultiPeak, 256 cells x 14 outputs, t = 0-26 s, linear phase plus
the onset of the nonlinear one). Everything for the pilot lives in this folder; paths are in `config.toml`.

| file | step |
|---|---|
| `regrid.py`, `test_regrid.py` | conservative regrid of a 120^3 cube onto a 32^3 box in v' = (v - V)/v_th (tested: mass, shape, outside fraction) |
| `build_dataset.py` | step 1: dataset npz (`g`, `n`, `V`, `vth`, `outside`, `N_v`, `cellid`, `x`, `out_idx`, `t`) |
| baselines, AE, evaluation | steps 2-4, not written yet |

```bash
python autoencoder/test_regrid.py
python autoencoder/build_dataset.py --max-files 2 --max-cells 3     # smoke test
python autoencoder/build_dataset.py                                 # full: 14 x 256 VDFs
```

Needs analysator and its dependencies (scipy, vtk, yt, ...) importable; `data_processing/vdf_tools.py` is reused for
reading and Hermite. Velocities are m/s in files, v' (dimensionless) in the dataset. g integrates to 1 - outside.
