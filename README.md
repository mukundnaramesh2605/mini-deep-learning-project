# Mini Deep Learning Project — MNIST Digit Classifier

A small end-to-end deep learning project: train an MLP on MNIST, track experiments with MLflow, and serve predictions through a Streamlit app where you can draw or upload a digit.

## Project structure

```
.
├── mini-deep-learning-project.ipynb          # Model exploration / training (MLP_MNIST, _2, _3, _4)
├── mini-deep-learning-project-mlflow.ipynb   # Training with MLflow experiment tracking
├── model.py                                  # MLP_MNIST_3 model definition (used by the app)
├── preprocess.py                             # Shared image preprocessing (crop, center, normalize)
├── app.py                                    # Streamlit app: draw or upload a digit, get a prediction
├── mnist_mlp.pt                              # Trained weights for MLP_MNIST_3
├── mlflow.db                                 # Local MLflow tracking store (SQLite)
├── mlartifacts/                              # MLflow logged model artifacts
├── dataset/                                  # MNIST data (downloaded via torchvision)
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` is a full environment export. The packages this project actually needs are: `torch`, `torchvision`, `numpy`, `pillow`, `streamlit`, `streamlit-drawable-canvas`, and `mlflow` (only for the tracking notebook).

## Model

`model.py` defines `MLP_MNIST_3`, a fully-connected network for 28x28 MNIST digits:

```
784 → 256 → ReLU → Dropout(p) → 128 → ReLU → Dropout(p) → 10
```

The notebook also explores smaller/larger variants (`MLP_MNIST`, `MLP_MNIST_2`, `MLP_MNIST_4`) to compare capacity and regularization tradeoffs. `MLP_MNIST_3` is the one used by the app, with weights saved in `mnist_mlp.pt`.

## Training

- `mini-deep-learning-project.ipynb` — trains and evaluates each model variant on MNIST using a standard PyTorch train/test loop.
- `mini-deep-learning-project-mlflow.ipynb` — same training flow, but logs params, metrics, and the model itself to MLflow, and registers the trained model in the MLflow Model Registry.

To view experiment runs, start the MLflow UI pointed at the local tracking store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

## Preprocessing

`preprocess.py` (mirrored inline in `app.py`) converts an arbitrary input image into an MNIST-style tensor:

1. Convert to grayscale.
2. Crop to the digit's bounding box and rescale to fit a 20x20 box.
3. Paste into a 28x28 canvas, centered by center of mass (like the original MNIST preprocessing).
4. Normalize to `[-1, 1]`.

## Running the app

```bash
streamlit run app.py
```

This opens a browser UI with two tabs:

- **Draw** — sketch a digit on a canvas and get a live prediction.
- **Upload** — upload an image of a digit (PNG/JPG).

The app shows the predicted digit, confidence, and a bar chart of the full probability distribution over 0–9.
