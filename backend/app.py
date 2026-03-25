import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=device))
model.eval()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = np.array(image).astype(np.float32) / 255.0

    mask = img > 0.1

    if not np.any(mask):
        img_resized = np.zeros((28, 28), dtype=np.float32)
    else:
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 20
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(img.shape[0] - 1, y_max + pad)
        x_max = min(img.shape[1] - 1, x_max + pad)

        img_cropped = img[y_min : y_max + 1, x_min : x_max + 1]

        pil_img = Image.fromarray((img_cropped * 255).astype(np.uint8))
        pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

        img_resized = np.array(pil_img).astype(np.float32) / 255.0

    img_resized = (img_resized - 0.1307) / 0.3081
    img_resized = (img_resized - 0.1307) / 0.3081

    # DEBUG (before normalization!)
    debug_img = ((img_resized * 0.3081 + 0.1307) * 255).astype(np.uint8)
    Image.fromarray(debug_img).save("debug_processed.png")

    tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


@app.get("/")
def root():
    return {"message": "MNIST backend is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        topk = torch.topk(probs, k=3)

        predictions = topk.indices[0].tolist()
        confidences = topk.values[0].tolist()

    return {
        "predictions": predictions,
        "confidences": [round(float(c), 4) for c in confidences],
    }
