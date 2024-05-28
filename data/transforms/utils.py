from torchvision import transforms as T


def get_interpolation_mode(mode: str):
    modes = {
        "bicubic": T.InterpolationMode.BICUBIC,
        "bilinear": T.InterpolationMode.BILINEAR,
        "box": T.InterpolationMode.BOX,
        "cubic": T.InterpolationMode.BICUBIC,
        "hamming": T.InterpolationMode.HAMMING,
        "lanczos": T.InterpolationMode.LANCZOS,
        "nearest": T.InterpolationMode.NEAREST,
    }

    if mode in modes:
        return modes[mode]
    else:
        raise ValueError(f"Interpolation mode {mode} is not valid.")
