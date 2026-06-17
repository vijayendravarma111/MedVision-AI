
# ==========================================================
# MEDVISION AI - BRAIN MRI UTILITIES
# ==========================================================

import os
import cv2
import torch
import numpy as np

from PIL import Image
from datetime import datetime

import torch.nn as nn

from torchvision import models
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# MODEL PATH
# ==========================================================

MODEL_PATH = "brain_model_complete.pth"

# ==========================================================
# IMAGE TRANSFORMS
# ==========================================================

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# LOAD MODEL
# ==========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280,4)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

# ==========================================================
# GRADCAM
# ==========================================================

target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# ==========================================================
# PREDICT MRI
# ==========================================================

def predict_mri(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224,224))

    input_tensor = test_transforms(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    pred_idx = torch.argmax(
        probabilities
    ).item()

    pred_class = class_names[pred_idx]

    confidence = (
        probabilities[pred_idx].item() * 100
    )

    probs = {}

    for idx, cls in enumerate(class_names):

        probs[cls] = round(
            probabilities[idx].item() * 100,
            2
        )

    return (
        pred_class,
        confidence,
        probs,
        pred_idx
    )

# ==========================================================
# GENERATE REPORT
# ==========================================================

def generate_report(
    prediction,
    confidence
):

    confidence = round(
        confidence,
        2
    )

    if prediction == "glioma":

        findings = (
            "MRI brain demonstrates imaging "
            "features suggestive of glioma."
        )

        recommendation = (
            "Neurology and neurosurgery "
            "consultation recommended."
        )

    elif prediction == "meningioma":

        findings = (
            "MRI brain demonstrates imaging "
            "features suggestive of meningioma."
        )

        recommendation = (
            "Specialist evaluation recommended."
        )

    elif prediction == "pituitary":

        findings = (
            "MRI brain demonstrates imaging "
            "features suggestive of pituitary tumor."
        )

        recommendation = (
            "Endocrinology consultation recommended."
        )

    else:

        findings = (
            "No significant imaging evidence "
            "of tumor detected."
        )

        recommendation = (
            "Routine follow-up recommended."
        )

    report = f"""
MEDVISION AI - BRAIN MRI REPORT

Findings:
{findings}

Impression:
{prediction.upper()} detected with {confidence:.2f}% confidence.

Recommendation:
{recommendation}

AI Generated Research Prototype Report
Not Intended For Clinical Diagnosis
"""

    return report

# ==========================================================
# GENERATE PDF
# ==========================================================

def create_pdf_report(
    prediction,
    confidence,
    report_text,
    output_path="outputs/reports/Brain_MRI_Report.pdf"
):

    os.makedirs(
        "outputs/reports",
        exist_ok=True
    )

    doc = SimpleDocTemplate(
        output_path
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "MEDVISION AI",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1,12)
    )

    elements.append(
        Paragraph(
            "Brain MRI Analysis Report",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(1,12)
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    elements.append(
        Paragraph(
            f"Generated: {timestamp}",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(1,12)
    )

    report_text = report_text.replace(
        "\n",
        "<br/>"
    )

    elements.append(
        Paragraph(
            report_text,
            styles["BodyText"]
        )
    )

    doc.build(elements)

    return output_path

# ==========================================================
# GENERATE GRADCAM
# ==========================================================

def generate_gradcam(
    image_path,
    pred_idx
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224,224)
    )

    image_np = np.array(
        image
    )

    rgb_img = image_np.astype(
        np.float32
    ) / 255.0

    input_tensor = test_transforms(
        image
    )

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    targets = [
        ClassifierOutputTarget(
            pred_idx
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs(
        "outputs/heatmaps",
        exist_ok=True
    )

    heatmap_path = (
        "outputs/heatmaps/gradcam.png"
    )

    cv2.imwrite(
        heatmap_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return heatmap_path

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("Model Loaded Successfully")

    print("Classes:")

    for c in class_names:

        print(c)

