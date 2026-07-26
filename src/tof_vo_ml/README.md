# tof-vo-ml

Train a small CNN to estimate the final odometry delta in an 8-frame ToF
distance window.

```bash
uv run python -m tof_vo_ml.train datasets/example.npz \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --output runs/tof_vo_model.pt
```

The input tensor shape is `(batch, 8, 8, 8)`: 8 consecutive normalized ToF
distance frames. The label is the last transition in the window,
`frame[6] -> frame[7]`, as `(delta_x, delta_y, delta_theta)`.

Datasets are loaded from `.npz` files exported by `tof_dataset` using
`distance_m`, `valid`, and `pose_xytheta`. If a same-name `.json` metadata file
exists, its `max_distance_m` is used for distance normalization; otherwise the
default is `4.0`.
