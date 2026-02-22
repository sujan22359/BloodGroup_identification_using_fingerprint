# app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import io
import os

# ==============================================================
# 🧩 Residual Block Definition
# ==============================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        out += identity
        return torch.relu(out)

# ==============================================================
# 🧠 Basic CNN Model with Residual Block
# ==============================================================
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            ResidualBlock(128),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==============================================================
# ⚙️ CONFIGURATION
# ==============================================================
MODEL_PATH = r"E:\Projects\Blood_Group_new\Blood_group_new\CNN\best_basic_cnn_with_residual.pth"
CLASS_NAMES = ["A-", "A+", "B-", "B+", "AB-", "AB+", "O-", "O+"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model():
    model = BasicCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ==============================================================
# 📦 Image Preprocessing
# ==============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    img = image.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    return input_tensor, rgb_img

# ==============================================================
# 🎨 Streamlit UI
# ==============================================================
st.set_page_config(page_title="Blood Group Prediction (Fingerprint)", layout="wide")

st.title("Fingerprint-Based Blood Group Prediction using Residual CNN + Grad-CAM")
st.markdown("""
This web app predicts a **person's blood group from a fingerprint image** using a 
**Convolutional Neural Network (CNN)** with **Residual Blocks** and explains the decision using **Grad-CAM (XAI)**.
""")

uploaded_file = st.file_uploader("📤 Upload a fingerprint image (.jpg, .png, .bmp)", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", width=300)

    # Process image
    input_tensor, rgb_img = preprocess_image(image)

    # Grad-CAM
    target_layers = [model.features[-2]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_label = CLASS_NAMES[pred_idx]
        confidence = probs[0, pred_idx].item()

    # Display results
    st.subheader("Prediction Result")
    st.markdown(f"**Predicted Blood Group:** `{predicted_label}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_img, caption="Original Fingerprint", use_container_width=True)
    with col2:
        st.image(cam_image, caption=f"Grad-CAM Heatmap ({predicted_label})", use_container_width=True)

    # Download Grad-CAM
    buf = io.BytesIO()
    plt.imsave(buf, cam_image)
    st.download_button(
        label="Download Grad-CAM Image",
        data=buf.getvalue(),
        file_name=f"GradCAM_{predicted_label}.png",
        mime="image/png"
    )

else:
    st.info("Upload a fingerprint image to get started.")

st.markdown("---")
st.markdown("© 2025 Blood Group Prediction App. All rights reserved.")
