import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .loss import *
    from .model import TactMorph2D
    from .preprocessing import preprocess_registration_image
except ImportError:
    from loss import *
    from model import TactMorph2D
    from preprocessing import preprocess_registration_image

try:
    import wandb
except ImportError as exc:
    raise ImportError(
        "wandb is required for this script. Install with `pip install wandb`."
    ) from exc

# select dir to images
IMAGE_DIR = "../captured_frames/"
# e.g. "fixed.png" or None for random fixed images
FIXED_IMAGE_NAME = "frame_000.jpg"
RESIZE_TO = None  # (width, height) or None to keep original size
MODEL_PREPROCESS = "maxpool"  # "none", "area", or "maxpool"
MODEL_INPUT_SIZE = (64, 64)  # (width, height), used when MODEL_PREPROCESS != "none"
checkpoint_path = "./ckpt"

batch_size = 256
epochs = 100
learning_rate = 1e-3
model_similarity_weight = 1.0
use_original_similarity_loss = False
original_similarity_weight = 1.0 - model_similarity_weight
# reg terms: for biotactip, use 5.0 and 1.0.
bending_weight = 5.0
argument_weight = 1.0
seed = 1313213
drop_last = True

WANDB_PROJECT = "tactmorph_biotactip"
WANDB_RUN_NAME = "tactmorph_biotactip"
HFLIP_PROB = 0.65
ROTATE_CHOICES = (0, 1, 3)  # 0, +90, -90 (k=3)


def load_gray(path, resize_to=None):
    # load image as grayscale, given file path.
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    if resize_to is not None:
        image = preprocess_registration_image(image, mode="none", size=resize_to)
    return image


def load_images_in_memory(
    image_dir,
    resize_to=None,
    preprocess_mode="none",
    model_input_size=None,
):
    # load images to a list
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [
        f
        for f in sorted(os.listdir(image_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        raise ValueError(f"No images found in {image_dir}")

    images = []
    original_images = []
    for fname in files:
        path = os.path.join(image_dir, fname)
        original_img = load_gray(path, resize_to=resize_to)
        if preprocess_mode != "none":
            model_img = preprocess_registration_image(
                original_img,
                mode=preprocess_mode,
                size=model_input_size,
            )
        else:
            model_img = original_img
        original_images.append((fname, original_img))
        images.append((fname, model_img))
    # inspection of image dimension
    if resize_to is None:
        h0, w0 = images[0][1].shape
        for name, img in images:
            if img.shape != (h0, w0):
                raise ValueError(
                    "Images have different sizes. Set RESIZE_TO to enforce a size."
                )
        h0, w0 = original_images[0][1].shape
        for name, img in original_images:
            if img.shape != (h0, w0):
                raise ValueError(
                    "Original images have different sizes. Set RESIZE_TO to enforce a size."
                )
    # get tensors list
    tensors = []
    original_tensors = []
    for name, img in images:
        # adding a leading dimension to form (1, H, W)
        tensors.append(torch.from_numpy(img)[None, ...])
    for name, img in original_images:
        original_tensors.append(torch.from_numpy(img)[None, ...])
    return files, tensors, original_tensors


def build_batch(moving_tensors, batch_indices, fixed_tensor=None):
    moving = torch.stack([moving_tensors[i] for i in batch_indices], dim=0)
    if fixed_tensor is not None:
        # creating a batch of identical fixed images
        fixed = fixed_tensor.expand(moving.shape[0], -1, -1, -1)
    else:
        # sample a batch of images
        rand_idx = [random.randint(0, len(moving_tensors) - 1)
                    for _ in batch_indices]
        fixed = torch.stack([moving_tensors[i] for i in rand_idx], dim=0)
    return moving, fixed


def _resize_to_square(img, size):
    # set to square, in order to do flip operations
    return F.interpolate(
        img.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def augment_pairs(moving, fixed, square_size=None, hflip_prob=0.5, rotate_choices=(0, 1, 3)):
    batch = moving.shape[0]
    new_moving = []
    new_fixed = []
    if square_size is None:
        # expecting [1, 2, H, W] tensor
        square_size = np.min(moving.size()[2:])
    for i in range(batch):
        m = moving[i]
        f = fixed[i]
        k = random.choice(rotate_choices)
        if k != 0:
            m = torch.rot90(m, k=k, dims=(1, 2))
            f = torch.rot90(f, k=k, dims=(1, 2))
        if random.random() < hflip_prob:
            m = torch.flip(m, dims=(2,))
            f = torch.flip(f, dims=(2,))

        m = _resize_to_square(m, square_size)
        f = _resize_to_square(f, square_size)
        new_moving.append(m)
        new_fixed.append(f)
    return torch.stack(new_moving, dim=0), torch.stack(new_fixed, dim=0)


def augment_pairs_with_original(
    moving,
    fixed,
    moving_original,
    fixed_original,
    hflip_prob=0.5,
    rotate_choices=(0, 1, 3),
):
    batch = moving.shape[0]
    model_square_size = int(np.min(moving.size()[2:]))
    original_square_size = int(np.min(moving_original.size()[2:]))

    new_moving = []
    new_fixed = []
    new_moving_original = []
    new_fixed_original = []

    for i in range(batch):
        m = moving[i]
        f = fixed[i]
        mo = moving_original[i]
        fo = fixed_original[i]

        k = random.choice(rotate_choices)
        do_flip = random.random() < hflip_prob

        if k != 0:
            m = torch.rot90(m, k=k, dims=(1, 2))
            f = torch.rot90(f, k=k, dims=(1, 2))
            mo = torch.rot90(mo, k=k, dims=(1, 2))
            fo = torch.rot90(fo, k=k, dims=(1, 2))

        if do_flip:
            m = torch.flip(m, dims=(2,))
            f = torch.flip(f, dims=(2,))
            mo = torch.flip(mo, dims=(2,))
            fo = torch.flip(fo, dims=(2,))

        new_moving.append(_resize_to_square(m, model_square_size))
        new_fixed.append(_resize_to_square(f, model_square_size))
        new_moving_original.append(_resize_to_square(mo, original_square_size))
        new_fixed_original.append(_resize_to_square(fo, original_square_size))

    return (
        torch.stack(new_moving, dim=0),
        torch.stack(new_fixed, dim=0),
        torch.stack(new_moving_original, dim=0),
        torch.stack(new_fixed_original, dim=0),
    )


def resize_flow_torch(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    src_h, src_w = flow.shape[-2:]
    dst_h, dst_w = size
    if (src_h, src_w) == (dst_h, dst_w):
        return flow

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale = torch.tensor([scale_x, scale_y], dtype=resized.dtype, device=resized.device)
    return resized * scale.view(1, 2, 1, 1)


def main():
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names, tensors, original_tensors = load_images_in_memory(
        IMAGE_DIR,
        resize_to=RESIZE_TO,
        preprocess_mode=MODEL_PREPROCESS,
        model_input_size=MODEL_INPUT_SIZE,
    )
    fixed_tensor = None
    fixed_original_tensor = None
    if FIXED_IMAGE_NAME is not None:
        try:
            fixed_index = names.index(FIXED_IMAGE_NAME)
        except ValueError as exc:
            raise ValueError(
                f"Fixed image '{FIXED_IMAGE_NAME}' not found in {IMAGE_DIR}"
            ) from exc
        fixed_tensor = tensors[fixed_index]
        fixed_original_tensor = original_tensors[fixed_index]

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "image_dir": IMAGE_DIR,
            "fixed_image_name": FIXED_IMAGE_NAME,
            "resize_to": RESIZE_TO,
            "model_preprocess": MODEL_PREPROCESS,
            "model_input_size": MODEL_INPUT_SIZE,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "model_similarity_weight": model_similarity_weight,
            "use_original_similarity_loss": use_original_similarity_loss,
            "original_similarity_weight": original_similarity_weight,
            "bending_weight": bending_weight,
            "argument_weight": argument_weight,
            "seed": seed,
        },
    )

    model = TactMorph2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    global_step = 0
    num_samples = len(tensors)
    for epoch in range(1, epochs + 1):
        perm = list(range(num_samples))
        random.shuffle(perm)
        if drop_last:
            last = (num_samples // batch_size) * batch_size
            perm = perm[:last]

        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_indices = perm[start: start + batch_size]
            if len(batch_indices) < batch_size:
                continue
            moving, fixed = build_batch(
                tensors, batch_indices, fixed_tensor=fixed_tensor)
            moving_original, fixed_original = build_batch(
                original_tensors, batch_indices, fixed_tensor=fixed_original_tensor)
            moving = moving.to(device)
            fixed = fixed.to(device)
            moving_original = moving_original.to(device)
            fixed_original = fixed_original.to(device)
            # obtain a batch
            moving, fixed, moving_original, fixed_original = augment_pairs_with_original(
                moving,
                fixed,
                moving_original,
                fixed_original,
                hflip_prob=HFLIP_PROB,
                rotate_choices=ROTATE_CHOICES,
            )

            warped, flow = model(moving, fixed)
            model_sim_loss = similarity_loss(fixed, warped, loss_type="NCC")
            if use_original_similarity_loss:
                flow_original = resize_flow_torch(flow, fixed_original.shape[-2:])
                warped_original = model.transformer(moving_original, flow_original)
                original_sim_loss = similarity_loss(fixed_original, warped_original, loss_type="NCC")
            else:
                original_sim_loss = torch.zeros((), dtype=model_sim_loss.dtype, device=device)
            bend_loss = bending_energy_loss(flow)
            arg_loss, mu = beltrami_loss(flow)
            loss = (
                model_sim_loss * model_similarity_weight
                + original_sim_loss * original_similarity_weight
                + bend_loss * bending_weight
                + arg_loss * argument_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/model_sim_loss": model_sim_loss,
                    "train/original_sim_loss": original_sim_loss,
                    "train/bending_loss": bend_loss,
                    "train/argument_loss": arg_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )
            global_step += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss,
                  "epoch": epoch}, step=global_step)
        if epoch % 20 == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(),
                       f"{checkpoint_path}/d_sight_tactmorph2d_{epoch}.pt")
            print(f"Saved checkpoint to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
