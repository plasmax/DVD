import glob
import re
from pathlib import Path

import numpy as np
import OpenEXR

try:
    import Imath
except ImportError:
    Imath = None


def linear_to_srgb(linear_rgb):
    linear_rgb = np.clip(linear_rgb, 0.0, None)
    threshold = 0.0031308
    low = linear_rgb * 12.92
    high = 1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055
    return np.where(linear_rgb <= threshold, low, high)


def _read_exr_legacy_rgb(path):
    if Imath is None:
        raise RuntimeError(
            "Imath is required for legacy OpenEXR InputFile API but is not installed."
        )

    exr_file = OpenEXR.InputFile(str(path))
    data_window = exr_file.header()["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr_file.header().get("channels", {})

    if all(c in channels for c in ("R", "G", "B")):
        r = np.frombuffer(exr_file.channel("R", float_type), dtype=np.float32)
        g = np.frombuffer(exr_file.channel("G", float_type), dtype=np.float32)
        b = np.frombuffer(exr_file.channel("B", float_type), dtype=np.float32)
        rgb = np.stack([r, g, b], axis=-1)
    elif "RGB" in channels:
        rgb = np.frombuffer(exr_file.channel("RGB", float_type), dtype=np.float32)
    else:
        raise ValueError(f"No RGB channels found in EXR: {path}")

    return rgb.reshape(height, width, 3)


def _read_exr_modern_rgb(path):
    exr_file = OpenEXR.File(str(path))
    part = exr_file.parts[0]

    if "RGB" in part.channels:
        rgb = np.asarray(part.channels["RGB"].pixels, dtype=np.float32)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return rgb

    channels = part.channels
    if all(c in channels for c in ("R", "G", "B")):
        r = np.asarray(channels["R"].pixels, dtype=np.float32)
        g = np.asarray(channels["G"].pixels, dtype=np.float32)
        b = np.asarray(channels["B"].pixels, dtype=np.float32)
        return np.stack([r, g, b], axis=-1)

    raise ValueError(f"No RGB channels found in EXR: {path}")


def read_exr_rgb(path):
    if hasattr(OpenEXR, "InputFile"):
        try:
            return _read_exr_legacy_rgb(path)
        except Exception:
            if hasattr(OpenEXR, "File"):
                return _read_exr_modern_rgb(path)
            raise
    return _read_exr_modern_rgb(path)


def _read_depth_legacy(path):
    if Imath is None:
        raise RuntimeError(
            "Imath is required for legacy OpenEXR InputFile API but is not installed."
        )

    exr_file = OpenEXR.InputFile(str(path))
    data_window = exr_file.header()["dataWindow"]
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)

    r = np.frombuffer(exr_file.channel("R", float_type), dtype=np.float32)
    return r.reshape(height, width)


def _read_depth_modern(path):
    exr_file = OpenEXR.File(str(path))
    part = exr_file.parts[0]
    channels = part.channels

    if "R" in channels:
        return np.asarray(channels["R"].pixels, dtype=np.float32)
    if "RGB" in channels:
        rgb = np.asarray(channels["RGB"].pixels, dtype=np.float32)
        if rgb.ndim == 3:
            return rgb[..., 0]
        return rgb
    raise ValueError(f"No R/RGB channels found in EXR: {path}")


def read_depth_exr(path):
    if hasattr(OpenEXR, "InputFile"):
        try:
            return _read_depth_legacy(path)
        except Exception:
            if hasattr(OpenEXR, "File"):
                return _read_depth_modern(path)
            raise
    return _read_depth_modern(path)


def save_depth_exr(output_path, depth):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"R": np.ascontiguousarray(depth).astype("f")}
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_path))


def save_rgb_depth_exr(output_path, depth):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = np.ascontiguousarray(depth).astype("f")
    zeros = np.zeros_like(depth)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {
        "R": depth,
        "G": zeros,
        "B": zeros,
    }
    with OpenEXR.File(header, channels) as exr_file:
        exr_file.write(str(output_path))


def _extract_frame_number(path):
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"No frame number found in filename: {path}")
    return int(matches[-1])


def gather_exr_sequence(input_pattern):
    input_path = Path(input_pattern)
    if input_path.is_dir():
        exr_files = sorted(input_path.glob("*.exr"))
    else:
        glob_pattern = re.sub(r"%\d*d", "*", input_pattern)
        matches = sorted(Path(p) for p in glob.glob(glob_pattern))
        if matches:
            exr_files = [p for p in matches if p.suffix.lower() == ".exr"]
        elif input_path.suffix.lower() == ".exr" and input_path.exists():
            exr_files = [input_path]
        else:
            exr_files = []

    if not exr_files:
        raise FileNotFoundError(
            "No EXR files found. Pass a directory, glob pattern, or a valid .exr file path."
        )

    return [(path, _extract_frame_number(path)) for path in exr_files]


def filter_frame_range(sequence, start_frame=None, end_frame=None):
    filtered = []
    for path, frame in sequence:
        if start_frame is not None and frame < start_frame:
            continue
        if end_frame is not None and frame > end_frame:
            continue
        filtered.append((path, frame))

    if not filtered:
        raise ValueError(
            f"No EXR frames matched the requested range start={start_frame}, end={end_frame}."
        )
    return filtered


def format_sequence_path(pattern, frame):
    try:
        return pattern % frame
    except TypeError:
        pass

    match = re.search(r"%0?(\d*)d", pattern)
    if match is None:
        raise ValueError(
            "Output path must contain a printf-style frame token such as %04d."
        )

    width = match.group(1)
    replacement = f"{frame:0{width}d}" if width else str(frame)
    return pattern[:match.start()] + replacement + pattern[match.end():]
