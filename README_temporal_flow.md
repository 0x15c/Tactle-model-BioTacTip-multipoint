# Temporal Residual Flow Prototype

This patch adds an isolated experimental pipeline for identity-free marker displacement estimation.

The simplified Option A model treats the previous dense flow field as memory:

```text
input  = [I0, It, u_{t-1,x}, u_{t-1,y}]
output = Δu_t
u_t    = u_{t-1} + Δu_t
```

The model is trained self-supervised by warping the reference image `I0` toward each current image `It`:

```text
loss = image_reconstruction + lambda_spatial * spatial_smoothness + lambda_temporal * residual_update_penalty
```

No explicit marker identity assignment is used. Marker identity is implicit through the reference coordinate system.

## Files

```text
temporal_flow/
    temporal_flow/
        __init__.py
        dataset.py          # sequence dataset and preprocessing
        model.py            # small U-Net residual flow model
        warp.py             # differentiable dense image warp
        losses.py           # Charbonnier, smoothness, photometric losses
        preprocess.py       # OpenCV crop/mask/camera preprocessing utilities
        viz.py              # flow and warp visualization utilities
        checkpoint.py       # checkpoint save/load helpers
    scripts/
        make_dataset_from_camera.py
        make_dataset_from_video.py
        train_temporal_option_a.py
        demo_realtime_temporal_flow.py
        eval_sequence.py
```

## Dataset layout

```text
dataset_root/
    seq_000001/
        ref.png
        frame_000000.png
        frame_000001.png
        frame_000002.png
        ...
    seq_000002/
        ref.png
        frame_000000.png
        ...
```

`ref.png` should be the no-contact or initial reference image. The `frame_*.png` files are the tactile frames in temporal order.

## Create dataset from camera

```bash
python temporal_flow/scripts/make_dataset_from_camera.py \
  --out-root data/temporal_flow \
  --camera 0 \
  --num-frames 300 \
  --width 350 \
  --height 350
```

Controls:

- `r`: capture/reset the reference frame
- `s`: start/stop recording sequence frames
- `n`: start a new sequence folder
- `q`: quit

## Create dataset from video

```bash
python temporal_flow/scripts/make_dataset_from_video.py \
  --video force_regression_test/Raw_Session_20260311_231504.avi \
  --out-root data/temporal_flow \
  --width 350 \
  --height 350 \
  --ref-mode first
```

## Train

```bash
python temporal_flow/scripts/train_temporal_option_a.py \
  --data-root data/temporal_flow \
  --epochs 50 \
  --batch-size 2 \
  --clip-length 8 \
  --width 350 \
  --height 350 \
  --save-dir temporal_flow/checkpoints
```

## Real-time demo

```bash
python temporal_flow/scripts/demo_realtime_temporal_flow.py \
  --checkpoint temporal_flow/checkpoints/best.pt \
  --camera 0 \
  --width 350 \
  --height 350
```

Controls:

- `r`: set current frame as reference and reset previous flow
- `q`: quit

## Evaluate one saved sequence

```bash
python temporal_flow/scripts/eval_sequence.py \
  --checkpoint temporal_flow/checkpoints/best.pt \
  --seq data/temporal_flow/seq_000001 \
  --out-dir temporal_flow/eval_seq_000001
```
