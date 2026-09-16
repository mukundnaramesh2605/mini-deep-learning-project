import numpy as np
import torch
from PIL import Image, ImageOps

def preprocess(img: Image.Image):
    img = ImageOps.grayscale(img)

    # crop to the drawn digit and fit it into a 20x20 box
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        w, h = img.size
        scale = 20 / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # paste into 28x28 centred by centre of mass, like MNIST
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    arr = np.array(canvas, dtype=np.float32)
    if arr.sum() > 0:
        cy, cx = np.array(np.nonzero(arr)).mean(axis=1)
        shift = (int(round(14 - cx)), int(round(14 - cy)))
        canvas = canvas.transform(canvas.size, Image.AFFINE, (1, 0, -shift[0], 0, 1, -shift[1]))
        arr = np.array(canvas, dtype=np.float32)

    arr = arr / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)