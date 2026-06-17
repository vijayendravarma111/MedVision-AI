# ============================================================
# BONEGUARD AI
# COMPLETE SINGLE FILE APP
# PART 1
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import cv2
import torch
import tempfile
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from datetime import datetime

import gradio as gr

from torchvision import transforms

import timm

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus.flowables import HRFlowable

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("BoneGuard AI Started")
print("=" * 60)
print("Using Device:", device)
print("=" * 60)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Fracture Detected",
    "Normal Bone"
]

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "boneguard_final.pth"

model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded Successfully")
print("=" * 60)

# ============================================================
# MEDICAL REPORT GENERATOR
# ============================================================

def generate_medical_report(
    prediction,
    confidence
):

    confidence = round(
        confidence,
        2
    )

    # ========================================================
    # FRACTURE
    # ========================================================

    if prediction == "Fracture Detected":

        report = f"""
BONE X-RAY REPORT
--------------------------------------------------

Prediction:
Fracture Detected

Confidence:
{confidence}%

Findings:
Radiographic examination demonstrates cortical
discontinuity and abnormal bone alignment.

Localized structural disruption is identified
within the affected bony region.

Impression:
Bone fracture detected.

Radiographic findings are consistent with
traumatic skeletal injury.

Recommendation:

- Orthopedic consultation recommended.

- Immobilization of affected region advised.

- Clinical correlation recommended.

- Further evaluation may be considered.
"""

    # ========================================================
    # NORMAL
    # ========================================================

    else:

        report = f"""
BONE X-RAY REPORT
--------------------------------------------------

Prediction:
Normal Bone

Confidence:
{confidence}%

Findings:
No obvious cortical disruption,
fracture line or acute osseous abnormality.

Bone alignment appears preserved.

Impression:
No radiographic evidence of fracture.

Recommendation:

- Routine clinical follow-up if symptoms persist.

- Additional imaging may be considered if clinically indicated.
"""

    return report
# ============================================================
# GRADCAM CLASS
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            self.save_activations
        )

        self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    # ========================================================
    # SAVE ACTIVATIONS
    # ========================================================

    def save_activations(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # ========================================================
    # SAVE GRADIENTS
    # ========================================================

    def save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # ========================================================
    # GENERATE CAM
    # ========================================================

    def generate_cam(
        self,
        input_tensor,
        target_class
    ):

        output = self.model(
            input_tensor
        )

        self.model.zero_grad()

        loss = output[
            0,
            target_class
        ]

        loss.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = torch.mean(
            gradients,
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(device)

        for i, w in enumerate(weights):

            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()

        cam /= (
            cam.max() + 1e-8
        )

        return cam.detach().cpu().numpy()

# ============================================================
# INITIALIZE GRADCAM
# ============================================================

target_layer = model.conv_head

gradcam = GradCAM(
    model,
    target_layer
)

print(
    "GradCAM Initialized Successfully"
)

# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def create_pdf_report(
    prediction,
    confidence,
    report_text
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    pdf_path = (
        f"Bone_Report_{timestamp}.pdf"
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    # ========================================================
    # TITLE
    # ========================================================

    title = Paragraph(
        "<b><font size=18>"
        "BoneGuard AI"
        "</font></b>",
        styles["Title"]
    )

    elements.append(title)

    subtitle = Paragraph(
        "<font size=12>"
        "AI-Powered Bone Fracture Report"
        "</font>",
        styles["BodyText"]
    )

    elements.append(subtitle)

    elements.append(
        Spacer(1, 10)
    )

    elements.append(
        HRFlowable(width="100%")
    )

    elements.append(
        Spacer(1, 10)
    )

    # ========================================================
    # DATE
    # ========================================================

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    elements.append(
        Paragraph(
            f"<b>Generated:</b> {current_time}",
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> {prediction}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    # ========================================================
    # REPORT
    # ========================================================

    report_text = report_text.replace(
        "•",
        "-"
    )

    lines = report_text.split(
        "\n"
    )

    for line in lines:

        if line.strip():

            elements.append(
                Paragraph(
                    line,
                    styles["BodyText"]
                )
            )

            elements.append(
                Spacer(1, 4)
            )

    elements.append(
        Spacer(1, 20)
    )

    footer = Paragraph(
        "<font size=10 color='gray'>"
        "Generated by BoneGuard AI Research Prototype"
        "</font>",
        styles["BodyText"]
    )

    elements.append(
        footer
    )

    doc.build(
        elements
    )

    return pdf_path

# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_bone_xray(
    input_image
):

    if input_image is None:

        return (
            "Please upload an X-ray image.",
            "",
            None,
            None
        )

    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_image = input_image.convert(
        "RGB"
    )

    resized_image = original_image.resize(
        (300, 300)
    )

    original_np = np.array(
        resized_image
    )

    # ========================================================
    # PREPROCESS
    # ========================================================

    image_tensor = transform(
        resized_image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)

    # ========================================================
    # PREDICTION
    # ========================================================

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        1
    )

    predicted_class = predicted_class.item()

    confidence = (
        confidence.item() * 100
    )

    prediction = class_names[
        predicted_class
    ]

    # ========================================================
    # REPORT
    # ========================================================

    report = generate_medical_report(
        prediction,
        confidence
    )

    # ========================================================
    # GRADCAM
    # ========================================================

    cam = gradcam.generate_cam(
        image_tensor,
        predicted_class
    )

    heatmap = cv2.resize(
        cam,
        (300, 300)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    # ========================================================
    # PDF
    # ========================================================

    pdf_path = create_pdf_report(
        prediction,
        confidence,
        report
    )

    # ========================================================
    # RESULT TEXT
    # ========================================================

    result_text = f"""
Prediction: {prediction}

Confidence: {confidence:.2f}%
"""

    return (
        result_text,
        report,
        Image.fromarray(
            overlay
        ),
        pdf_path
    )
# ============================================================
# PROFESSIONAL UI CSS
# ============================================================

custom_css = """

body {
    background-color: #050B1E;
}

.gradio-container {
    background: #050B1E !important;
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #B8C1EC;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    background: #11182D;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #2A3452;
}

.footer-text {
    text-align: center;
    color: gray;
    margin-top: 20px;
}
"""

# ============================================================
# CREATE APPLICATION
# ============================================================

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft()
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML("""
    <div class="main-title">
        BoneGuard AI
    </div>

    <div class="sub-title">
        AI-Powered Bone Fracture Detection &
        Automated Radiology Report Generation
    </div>
    """)

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================


    # ========================================================
    # MAIN CONTENT
    # ========================================================

    with gr.Row():

        # ====================================================
        # LEFT SIDE
        # ====================================================

        with gr.Column(scale=1):

            gr.HTML("""
            <div class="result-box">
            <h2>Upload Bone X-Ray</h2>

            <p>
            Upload a Bone X-Ray image for fracture analysis.
            </p>

            </div>
            """)

            input_image = gr.Image(
                type="pil",
                label="Bone X-Ray",
                height=400
            )

            analyze_btn = gr.Button(
                "Analyze X-Ray",
                variant="primary",
                size="lg"
            )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        with gr.Column(scale=1):

            prediction_output = gr.Textbox(
                label="Prediction Result",
                lines=4
            )

            report_output = gr.Textbox(
                label="Medical Report",
                lines=18
            )

    # ========================================================
    # GRADCAM SECTION
    # ========================================================

    gr.Markdown(
        "## AI Attention Heatmap (GradCAM)"
    )

    gradcam_output = gr.Image(
        label="GradCAM Visualization",
        height=450
    )

    # ========================================================
    # PDF DOWNLOAD
    # ========================================================

    pdf_output = gr.File(
        label="Download PDF Report"
    )

    # ========================================================
    # BUTTON ACTION
    # ========================================================

    analyze_btn.click(

        fn=analyze_bone_xray,

        inputs=input_image,

        outputs=[
            prediction_output,
            report_output,
            gradcam_output,
            pdf_output
        ]
    )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div class="footer-text">
        © 2026 BoneGuard AI Research Prototype
    </div>
    """)

# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":

    app.launch(
        share=False,
        debug=False
    )