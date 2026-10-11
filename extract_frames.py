#!/usr/bin/env python3
"""
Extract sequential frames from the Bad Apple video for DNN training.

Features:
- Supports extraction via ffmpeg (fastest, lossless/high-quality PNG or JPEG)
- Configurable output directory, format, image dimensions, grayscale/RGB
- Sequential zero-padded file naming (e.g., frame_000000.png) suitable for PyTorch Dataset sorting
- Summary of extracted frame count and dimensions
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_default_video(workspace_dir: Path) -> Path | None:
    """Find the Bad Apple video in workspace."""
    candidates = list(workspace_dir.glob("*Bad Apple*.mp4")) + list(workspace_dir.glob("*.mp4"))
    return candidates[0] if candidates else None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    img_format: str = "png",
    resize: tuple[int, int] | None = None,
    grayscale: bool = False,
    start_index: int = 0,
    zero_pad: int = 6,
) -> int:
    """
    Extract frames sequentially from a video file using ffmpeg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    
    vf_filters = []
    if grayscale:
        vf_filters.append("format=gray")
    if resize:
        vf_filters.append(f"scale={resize[0]}:{resize[1]}")

    vf_arg = []
    if vf_filters:
        vf_arg = ["-vf", ",".join(vf_filters)]

    
    pattern = f"frame_%0{zero_pad}d.{img_format}"
    output_pattern = str(output_dir / pattern)

    cmd = [
        "ffmpeg",
        "-y",  
        "-i", str(video_path),
        *vf_arg,
        "-start_number", str(start_index),
    ]

    if img_format == "png":
        cmd += ["-compression_level", "3"]  
    elif img_format in ("jpg", "jpeg"):
        cmd += ["-q:v", "2"]  

    cmd.append(output_pattern)

    print(f"[+] Extracting frames from: {video_path.name}")
    print(f"[+] Output directory: {output_dir}")
    print(f"[+] Frame pattern: {pattern} (start index: {start_index})")
    if resize:
        print(f"[+] Resizing to: {resize[0]}x{resize[1]}")
    if grayscale:
        print("[+] Color mode: Grayscale")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[-] ffmpeg error:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("ffmpeg extraction failed")

    extracted_files = list(output_dir.glob(f"frame_*.{img_format}"))
    count = len(extracted_files)
    print(f"[✓] Successfully extracted {count} frames to {output_dir}")
    return count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sequential frames from video for DNN training."
    )
    parser.add_argument(
        "--video",
        "-i",
        type=Path,
        default=None,
        help="Path to input video file (defaults to Bad Apple video in current dir).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("frames"),
        help="Directory where frames will be saved (default: ./frames).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Image format for extracted frames (default: png).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize (width height), e.g. --resize 256 192 or --resize 512 384.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert frames to 1-channel grayscale.",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=6,
        help="Number of zero-padding digits in filename (default: 6 -> frame_000000.png).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for sequential filenames (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    workspace_dir = Path(__file__).resolve().parent
    video_path = args.video
    if video_path is None:
        video_path = find_default_video(workspace_dir)
        if video_path is None:
            print("[-] No video file found in workspace. Specify with --video /path/to/video.mp4", file=sys.stderr)
            sys.exit(1)

    if not video_path.exists():
        print(f"[-] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    resize = tuple(args.resize) if args.resize else None
    extract_frames(
        video_path=video_path,
        output_dir=args.output,
        img_format=args.format,
        resize=resize,
        grayscale=args.grayscale,
        start_index=args.start_index,
        zero_pad=args.zero_pad,
    )


if __name__ == "__main__":
    main()
