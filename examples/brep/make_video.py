"""
Makes the video from the frames in frames/, in order, each step shown for the same time, the junction moves between them faster:

1. a GIF, for the README: plays inline on GitHub and on PyPI;
2. an MP4 (H.264), for LinkedIn, which plays native video in the feed but may post an animated GIF as a still image.

Run after generation/extract_contours.py (frame 00) and draw_frames.py (the other frames).
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np
from PIL import Image
import imageio.v2 as imageio

FRAMES_DIR = HERE / "frames"
GIF_PATH = HERE / "brep_demo.gif"
MP4_PATH = HERE / "brep_demo.mp4"
SECONDS_PER_FRAME = 1
SECONDS_PER_MOVE = 0.2  # the frames of a junctions step's moves, one per point: draw_frames.py names them *_moving.png
MP4_FPS = 25  # a fixed frame rate: each frame repeated for its duration


def make_video():
    paths = sorted(FRAMES_DIR.glob("*.png"))
    frames = [Image.open(path).convert("RGB") for path in paths]
    durations = [SECONDS_PER_MOVE if path.stem.endswith("_moving") else SECONDS_PER_FRAME for path in paths]
    write_gif(frames, durations, GIF_PATH)
    write_mp4(frames, durations, MP4_PATH)


def write_gif(frames: list[Image.Image], durations: list[float], path) -> None:
    """With Pillow: the frames' flat colors fit in the GIF's 256."""
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[round(seconds * 1000) for seconds in durations], loop=0)
    print(f"Saved {path}")


def write_mp4(frames: list[Image.Image], durations: list[float], path) -> None:
    """600 px is not a multiple of 16: macro_block_size=8, or imageio scales the frames up to 608."""
    with imageio.get_writer(path, fps=MP4_FPS, codec="libx264", macro_block_size=8) as writer:
        for frame, seconds in zip(frames, durations):
            image = np.asarray(frame)
            for _ in range(round(seconds * MP4_FPS)):
                writer.append_data(image)
    print(f"Saved {path}")


if __name__ == "__main__":
    make_video()
