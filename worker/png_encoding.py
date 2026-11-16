"""Lossless worker PNG modes; candidate pixels and output size stay fixed."""
import base64
import io

from PIL import Image


MODES = ("legacy", "fast")


def encode_png(sprite, scale, mode="legacy"):
    if mode not in MODES:
        raise ValueError("PNG mode must be legacy or fast")
    image = Image.fromarray(sprite).resize((16 * scale, 16 * scale), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    if mode == "legacy":
        image.save(buffer, "PNG", optimize=True)
    else:
        image.save(buffer, "PNG", optimize=False, compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def pixels(data_url):
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
        image.load()
        return image.convert("RGBA")
