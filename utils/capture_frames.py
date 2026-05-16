import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


# ---------------- Configuration ----------------

@dataclass
class Config:
    # Device / IO
    input_source: int | str = 0
    output_dir: str = "captured_frames"
    prefix: str = "frame"
    num_images: int = 100

    # Image geometry, adapted from your original script
    image_size: int = 350
    radius: int = 160
    center: tuple = (175, 175)

    crop_px: int = 175
    crop_py: int = 175
    crop_offset_x: int = 0
    crop_offset_y: int = -8

    # Camera properties, adapted from your original script
    exposure: float = -7.8
    brightness: float = 0
    contrast: float = 64
    saturation: float = 60
    hue: float = 0
    gain: float = 0

    # Save settings
    save_processed: bool = True
    save_raw_crop: bool = False
    jpg_quality: int = 95


# ---------------- Preprocessing helpers ----------------

def create_radial_mask(size, center=None, max_value=245, power=2):
    """
    Illumination correction mask.
    This follows the idea from your original script:
    darker / stronger correction near the edge.
    """
    h, w = (size, size) if isinstance(size, int) else size

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    normalized = distance / np.sqrt(cx ** 2 + cy ** 2)

    mask = (normalized ** power) * max_value
    return np.clip(mask, 0, max_value).astype(np.uint8)


def compute_crop_limits(frame_width, frame_height, cfg: Config):
    cx = int(frame_width / 2)
    cy = int(frame_height / 2)

    x1 = cx - cfg.crop_px + cfg.crop_offset_x
    y1 = cy - cfg.crop_py + cfg.crop_offset_y
    x2 = cx + cfg.crop_px + cfg.crop_offset_x
    y2 = cy + cfg.crop_py + cfg.crop_offset_y

    # Clamp to valid image region
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    return x1, y1, x2, y2


def preprocess_frame(frame, cfg: Config, crop_limits, circular_mask, radial_mask):
    x1, y1, x2, y2 = crop_limits

    frame_cropped = frame[y1:y2, x1:x2]

    # If crop size does not perfectly match the mask size, resize to expected size.
    expected_size = (2 * cfg.crop_px, 2 * cfg.crop_py)

    if frame_cropped.shape[1] != expected_size[0] or frame_cropped.shape[0] != expected_size[1]:
        frame_cropped = cv.resize(frame_cropped, expected_size)

    # Apply circular mask
    masked = cv.bitwise_and(frame_cropped, frame_cropped, mask=circular_mask)

    # Convert to grayscale
    gray = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

    # Illumination correction, same idea as your original script
    corrected = cv.subtract(gray, radial_mask)

    return corrected, frame_cropped


# ---------------- Main capture loop ----------------

def main(cfg: Config):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(cfg.input_source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {cfg.input_source}")

    # Read current camera size
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Apply camera settings from your original script
    cap.set(cv.CAP_PROP_EXPOSURE, cfg.exposure)
    cap.set(cv.CAP_PROP_BRIGHTNESS, cfg.brightness)
    cap.set(cv.CAP_PROP_CONTRAST, cfg.contrast)
    cap.set(cv.CAP_PROP_SATURATION, cfg.saturation)
    cap.set(cv.CAP_PROP_HUE, cfg.hue)
    cap.set(cv.CAP_PROP_GAIN, cfg.gain)

    crop_w = 2 * cfg.crop_px
    crop_h = 2 * cfg.crop_py

    crop_limits = compute_crop_limits(frame_width, frame_height, cfg)

    # Circular mask
    circular_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv.circle(circular_mask, cfg.center, cfg.radius, 255, -1)

    # Radial illumination correction mask
    radial_mask = create_radial_mask(
        size=(crop_h, crop_w),
        center=cfg.center,
        max_value=245,
    )

    print("Camera opened.")
    print(f"Saving to: {output_dir.resolve()}")
    print(f"Target number of images: {cfg.num_images}")
    print()
    print("Controls:")
    print("  s : start / pause saving")
    print("  q : quit")
    print("  r : reset image counter")
    print()

    saving = False
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        processed, raw_crop = preprocess_frame(
            frame,
            cfg,
            crop_limits,
            circular_mask,
            radial_mask,
        )

        # Display status text
        preview = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)

        status = "SAVING" if saving else "PAUSED"
        cv.putText(
            preview,
            f"{status} | {count}/{cfg.num_images}",
            (10, 25),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if saving else (0, 0, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            preview,
            "s: start/pause | q: quit | r: reset",
            (10, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

        cv.imshow("processed preview", preview)
        cv.imshow("raw camera", frame)

        if cfg.save_raw_crop:
            cv.imshow("raw cropped preview", raw_crop)

        # Save current frame if enabled
        if saving and count < cfg.num_images:
            filename = f"{cfg.prefix}_{count:03d}.jpg"

            if cfg.save_processed:
                save_path = output_dir / filename
                cv.imwrite(
                    str(save_path),
                    processed,
                    [cv.IMWRITE_JPEG_QUALITY, cfg.jpg_quality],
                )

            if cfg.save_raw_crop:
                raw_filename = f"{cfg.prefix}_{count:03d}_raw.jpg"
                raw_save_path = output_dir / raw_filename
                cv.imwrite(
                    str(raw_save_path),
                    raw_crop,
                    [cv.IMWRITE_JPEG_QUALITY, cfg.jpg_quality],
                )

            print(f"Saved image {count}: {filename}")
            count += 1

        if count >= cfg.num_images:
            print("Finished saving target number of images.")
            break

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit by user.")
            break

        elif key == ord("s"):
            saving = not saving
            print("Saving started." if saving else "Saving paused.")

        elif key == ord("r"):
            count = 0
            print("Counter reset to 0.")

    cap.release()
    cv.destroyAllWindows()


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture camera frames and save them as images."
    )

    parser.add_argument(
        "--input",
        default="0",
        help="Camera index or video file path. Default: 0",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="captured_frames",
        help="Folder to save images.",
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images to save.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Output image filename prefix.",
    )

    parser.add_argument(
        "--save-raw-crop",
        action="store_true",
        help="Also save the raw cropped BGR image.",
    )

    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100.",
    )

    args = parser.parse_args()

    # Convert input to int if it looks like a camera index
    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input

    cfg = Config(
        input_source=input_source,
        output_dir=args.output_dir,
        num_images=args.num_images,
        prefix=args.prefix,
        save_raw_crop=args.save_raw_crop,
        jpg_quality=args.jpg_quality,
    )

    main(cfg)