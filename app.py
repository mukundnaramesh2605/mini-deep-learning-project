import streamlit as st
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from model import MLP_MNIST_3

st.set_page_config(page_title="MNIST Digit Classifier")
st.title("MNIST Digit Classifier")

@st.cache_resource
def load_model():
    m = MLP_MNIST_3(hidden=256, p=0.3)
    m.load_state_dict(torch.load("mnist_mlp.pt", map_location="cpu"))
    m.eval()
    return m

model = load_model()

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

tab1, tab2 = st.tabs(["Draw", "Upload"])

with tab1:
    canvas = st_canvas(fill_color="white", stroke_width=18, stroke_color="white",
                       background_color="black", width=280, height=280,
                       return_image_data=True, key="canvas")
    data = canvas.image_data
    if data is not None and data[:, :, :3].sum() > 0:
        img = Image.fromarray(data.astype("uint8"))
        x = preprocess(img)

with tab2:
    file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=150)
        x = preprocess(ImageOps.invert(ImageOps.grayscale(img)) if np.array(img).mean() > 127 else img)

if "x" in locals():
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    pred = probs.argmax().item()
    st.subheader(f"Prediction: {pred}")
    st.caption(f"Confidence: {probs[pred]:.2%}")
    st.bar_chart({str(i): float(p) for i, p in enumerate(probs)})