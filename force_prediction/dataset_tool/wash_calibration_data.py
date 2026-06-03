import argparse
import json
import re
from pathlib import Path

import cv2 as cv


FORCE_PATTERN = re.compile(
    r"Fx:\s*(?P<Fx>[-+]?\d*\.?\d+)\s*,\s*"
    r"Fy:\s*(?P<Fy>[-+]?\d*\.?\d+)\s*,\s*"
    r"Fz:\s*(?P<Fz>[-+]?\d*\.?\d+)"
)
STAMP_PATTERN = re.compile(r"(?P<stamp>\d{8}_\d{6}_\d+)")


def extract_stamp(path: Path) -> str:
    match = STAMP_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse timestamp from filename: {path.name}")
    return match.group("stamp")


def parse_force_txt(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = FORCE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not parse Fx/Fy/Fz from {path}")
    return {key: float(value) for key, value in match.groupdict().items()}


def center_crop_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def crop_resize_image(src_path: Path, dst_path: Path, size: int) -> None:
    image = cv.imread(str(src_path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {src_path}")
    image = center_crop_square(image)
    image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(dst_path), image)


def sorted_init_frames(init_dir: Path) -> list[tuple[str, Path]]:
    frames = []
    for path in sorted(init_dir.glob("init_raw_*.png")):
        frames.append((extract_stamp(path), path))
    if not frames:
        raise ValueError(f"No init_raw_*.png files found in {init_dir}")
    return frames


def choose_init_frame(sample_stamp: str, init_frames: list[tuple[str, Path]]) -> tuple[str, Path]:
    chosen = init_frames[0]
    for stamp, path in init_frames:
        if stamp <= sample_stamp:
            chosen = (stamp, path)
        else:
            break
    return chosen


def wash_dataset(input_dir: Path, output_dir: Path, image_size: int) -> tuple[int, int]:
    data_dir = input_dir / "data"
    raw_dir = input_dir / "raw_frames"
    init_dir = input_dir / "init_frames"

    out_raw_dir = output_dir / "raw_frames"
    out_init_dir = output_dir / "init_frames"
    manifest_path = output_dir / "labels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_frames = sorted_init_frames(init_dir)
    processed_inits = set()

    total = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for label_path in sorted(data_dir.glob("data_*.txt")):
            try:
                stamp = extract_stamp(label_path)
                force = parse_force_txt(label_path)
                raw_path = raw_dir / f"raw_{stamp}.png"
                if not raw_path.exists():
                    raise FileNotFoundError(f"Missing raw frame for {label_path.name}: {raw_path}")

                init_stamp, init_path = choose_init_frame(stamp, init_frames)

                washed_raw = out_raw_dir / f"raw_{stamp}.png"
                washed_init = out_init_dir / f"init_raw_{init_stamp}.png"
                crop_resize_image(raw_path, washed_raw, image_size)
                if init_stamp not in processed_inits:
                    crop_resize_image(init_path, washed_init, image_size)
                    processed_inits.add(init_stamp)

                record = {
                    "sample_id": stamp,
                    "raw_image": str(washed_raw.relative_to(output_dir)).replace("\\", "/"),
                    "init_image": str(washed_init.relative_to(output_dir)).replace("\\", "/"),
                    "label_file": str(label_path.relative_to(input_dir)).replace("\\", "/"),
                    "force": force,
                }
                manifest.write(json.dumps(record) + "\n")
                total += 1
            except Exception as exc:
                print(f"Skipping {label_path.name}: {exc}")
                skipped += 1

    return total, skipped


def main():
    parser = argparse.ArgumentParser(description="Wash force calibration txt labels and matching frames into JSONL.")
    parser.add_argument("--input", type=Path, default=Path("force_prediction/calibration_data"))
    parser.add_argument("--output", type=Path, default=Path("force_prediction/calibration_data_washed"))
    parser.add_argument("--image-size", type=int, default=360)
    args = parser.parse_args()

    total, skipped = wash_dataset(args.input, args.output, args.image_size)
    print(f"Wrote {total} records to {args.output / 'labels.jsonl'}")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    main()
