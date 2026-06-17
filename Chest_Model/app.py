# app.py (PART A)

# ==========================================================
# MEDVISION AI
# CHEST DISEASE DETECTION SYSTEM
# PART A
# ==========================================================

import os
import cv2
import torch
import numpy as np
import streamlit as st

from PIL import Image
from datetime import datetime

import torch.nn as nn
from torchvision import models
from torchvision import transforms

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="MEDVISION AI",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main{
        background-color:#0e1117;
    }

    .title{
        text-align:center;
        font-size:42px;
        font-weight:bold;
        color:#00d4ff;
    }

    .subtitle{
        text-align:center;
        color:white;
        font-size:18px;
        margin-bottom:20px;
    }

    .prediction-box{
        padding:20px;
        border-radius:15px;
        background:#1f2937;
        color:white;
    }

    .report-box{
        padding:20px;
        border-radius:15px;
        background:#111827;
        color:white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

TEST_TRANSFORM = transforms.Compose([

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

@st.cache_resource
def load_model():

    model = models.densenet121(
        weights=None
    )

    model.classifier = nn.Linear(
        1024,
        2
    )

    model.load_state_dict(
        torch.load(
            "best_chest_model_v2.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    model.to(DEVICE)

    return model

# ==========================================================
# LOAD MODEL INSTANCE
# ==========================================================

model = load_model()

# ==========================================================
# PREPROCESS IMAGE
# ==========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    tensor = TEST_TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_xray(image):

    input_tensor = preprocess_image(
        image
    )

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probs,
            dim=1
        )

    predicted_class = CLASS_NAMES[
        prediction.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    return (
        predicted_class,
        confidence_score,
        input_tensor
    )

# ==========================================================
# SMART REPORT GENERATOR
# ==========================================================

def generate_medical_report(
    prediction,
    confidence
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if prediction == "NORMAL":

        findings = (
            "No focal lung opacity detected. "
            "Lung fields appear clear. "
            "No significant radiographic abnormalities."
        )

        impression = (
            "No evidence of pneumonia detected."
        )

        recommendation = (
            "Routine clinical follow-up if symptoms persist."
        )

    else:

        findings = (
            "Patchy radiographic opacity identified "
            "within the lung fields. Findings may be "
            "consistent with pneumonia."
        )

        impression = (
            "AI analysis suggests possible pneumonia."
        )

        recommendation = (
            "Clinical correlation and radiologist review recommended."
        )

    return {

        "timestamp": timestamp,

        "prediction": prediction,

        "confidence": round(
            confidence,
            2
        ),

        "findings": findings,

        "impression": impression,

        "recommendation": recommendation

    }

# ==========================================================
# PDF REPORT GENERATOR
# ==========================================================

def create_pdf_report(report):

    pdf_path = "reports/MedVision_Report.pdf"

    os.makedirs(
        "reports",
        exist_ok=True
    )

    doc = SimpleDocTemplate(
        pdf_path
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "MEDVISION AI",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            "AI Medical Imaging Report",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(1,20)
    )

    content.append(
        Paragraph(
            f"<b>Timestamp:</b> {report['timestamp']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Prediction:</b> {report['prediction']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence:</b> {report['confidence']}%",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1,15)
    )

    content.append(
        Paragraph(
            "<b>Findings</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            report["findings"],
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1,15)
    )

    content.append(
        Paragraph(
            "<b>Impression</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            report["impression"],
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1,15)
    )

    content.append(
        Paragraph(
            "<b>Recommendation</b>",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            report["recommendation"],
            styles["BodyText"]
        )
    )

    doc.build(content)

    return pdf_path

# ==========================================================
# GRADCAM FUNCTION
# ==========================================================

def generate_gradcam(
    image,
    input_tensor
):

    target_layers = [
        model.features[-1]
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    rgb_img = np.array(
        image.resize((224,224))
    ).astype(np.float32) / 255.0

    targets = [
        ClassifierOutputTarget(1)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs(
        "gradcam_outputs",
        exist_ok=True
    )

    output_path = (
        "gradcam_outputs/gradcam_result.jpg"
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )

    return (
        visualization,
        output_path
    )

# ==========================================================
# PART A END
# ==========================================================

# app.py (PART B)


# ==========================================================
# MEDVISION AI
# PART B
# STREAMLIT DASHBOARD UI
# ==========================================================

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.markdown(
    """
    <div class="title">
        🩺 MEDVISION AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Powered Chest Disease Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------

with st.sidebar:

    st.title("MEDVISION AI")

    st.info(
        """
        Upload a Chest X-Ray image.

        The system will:

        ✔ Predict Disease

        ✔ Show Confidence

        ✔ Generate GradCAM

        ✔ Generate Smart Report

        ✔ Download PDF Report
        """
    )

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["png", "jpg", "jpeg"]
)

# ----------------------------------------------------------
# PROCESS IMAGE
# ----------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            width=350
        )

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    prediction, confidence, input_tensor = predict_xray(
        image
    )

    report = generate_medical_report(
        prediction,
        confidence
    )

    with col2:

        st.subheader("Prediction")

        if prediction == "NORMAL":

            st.success(
                f"Prediction: {prediction}"
            )

        else:

            st.error(
                f"Prediction: {prediction}"
            )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    st.divider()

    # ------------------------------------------------------
    # GRADCAM
    # ------------------------------------------------------

    st.subheader(
        "Explainable AI (GradCAM)"
    )

    gradcam_image, gradcam_path = generate_gradcam(
        image,
        input_tensor
    )

    grad_col1, grad_col2 = st.columns(2)

    with grad_col1:

        st.image(
            image,
            caption="Original X-Ray",
            use_container_width=350
        )

    with grad_col2:

        st.image(
            gradcam_image,
            caption="GradCAM Heatmap",
            width=350
        )

    st.divider()

    # ------------------------------------------------------
    # REPORT SECTION
    # ------------------------------------------------------

    st.subheader(
        "Smart Clinical Report"
    )

    with st.container():

        st.markdown(
            f"""
            ### Timestamp
            {report['timestamp']}

            ### Prediction
            {report['prediction']}

            ### Confidence
            {report['confidence']}%

            ### Findings
            {report['findings']}

            ### Impression
            {report['impression']}

            ### Recommendation
            {report['recommendation']}
            """
        )

    st.divider()

    # ------------------------------------------------------
    # PDF GENERATION
    # ------------------------------------------------------

    pdf_path = create_pdf_report(
        report
    )

    with open(
        pdf_path,
        "rb"
    ) as pdf_file:

        st.download_button(

            label="📄 Download PDF Report",

            data=pdf_file,

            file_name="MedVision_AI_Report.pdf",

            mime="application/pdf"
        )

    # ------------------------------------------------------
    # GRADCAM DOWNLOAD
    # ------------------------------------------------------

    with open(
        gradcam_path,
        "rb"
    ) as cam_file:

        st.download_button(

            label="🔥 Download GradCAM",

            data=cam_file,

            file_name="GradCAM_Result.jpg",

            mime="image/jpeg"
        )

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        margin-top:20px;
        color:#9CA3AF;
        font-size:13px;
        background:#111827;
        border-radius:10px;
    ">
        🩺 MEDVISION AI |
        Chest Disease Detection System |
        Powered by DenseNet121 + GradCAM |
        Research Prototype v1.0
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# END OF APP.PY
# ==========================================================
