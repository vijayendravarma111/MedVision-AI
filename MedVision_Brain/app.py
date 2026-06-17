
# ==========================================================
# MEDVISION AI - BRAIN MRI SYSTEM
# ==========================================================

import gradio as gr
from PIL import Image

from brain_utils import (
    predict_mri,
    generate_report,
    generate_gradcam,
    create_pdf_report
)

# ==========================================================
# MAIN FUNCTION
# ==========================================================

def analyze_mri(image):

    if image is None:

        return (
            "No Image Uploaded",
            {},
            "",
            None,
            None
        )

    image_path = "temp_mri.jpg"

    image.save(image_path)

    prediction, confidence, probs, pred_idx = predict_mri(
        image_path
    )

    report = generate_report(
        prediction,
        confidence
    )

    heatmap_path = generate_gradcam(
        image_path,
        pred_idx
    )

    pdf_path = create_pdf_report(
        prediction,
        confidence,
        report
    )

    prediction_text = f"""
Prediction : {prediction}

Confidence : {confidence:.2f}%
"""

    return (
        prediction_text,
        probs,
        report,
        heatmap_path,
        pdf_path
    )

# ==========================================================
# DARK MEDICAL THEME
# ==========================================================

css = """

.gradio-container {
    background-color: #07111F !important;
}

h1 {
    text-align:center;
    color:white;
}

.dark-box {
    background:#111827;
    padding:15px;
    border-radius:12px;
}

"""

# ==========================================================
# UI
# ==========================================================

with gr.Blocks(
    css=css,
    title="MedVision AI"
) as demo:

    gr.Markdown(
    """
# 🧠 MedVision AI
### Brain MRI Tumor Detection System
"""
    )

    with gr.Row():

        # LEFT PANEL

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Upload Brain MRI"
            )

            analyze_button = gr.Button(
                "Analyze MRI",
                variant="primary"
            )

        # RIGHT PANEL

        with gr.Column(scale=1):

            prediction_output = gr.Textbox(
                label="Prediction Result",
                lines=4
            )

            probability_output = gr.Label(
                label="Class Probabilities"
            )

    # REPORT

    report_output = gr.Textbox(
        label="Clinical Report",
        lines=15
    )

    # GRADCAM

    heatmap_output = gr.Image(
        label="GradCAM Visualization"
    )

    # PDF

    pdf_output = gr.File(
        label="Download PDF Report"
    )

    analyze_button.click(
        fn=analyze_mri,
        inputs=input_image,
        outputs=[
            prediction_output,
            probability_output,
            report_output,
            heatmap_output,
            pdf_output
        ]
    )

# ==========================================================
# LAUNCH
# ==========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )

