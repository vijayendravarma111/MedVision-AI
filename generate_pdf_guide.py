import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_project_guide():
    pdf_path = "MedVision_AI_Project_Guide.pdf"
    
    # Establish document margins and layout
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom clean styling parameters
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'), # Charcoal / Slate
        alignment=1, # Centered
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'), # Muted grey
        alignment=1,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1E3A8A'), # Navy blue
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0D9488'), # Teal
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    q_style = ParagraphStyle(
        'Question_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    a_style = ParagraphStyle(
        'Answer_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        leftIndent=10,
        spaceAfter=10
    )

    story = []
    
    # ----------------------------------------------------
    # TITLE & HEADER
    # ----------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("MEDVISION AI: ARCHITECTURAL DEEP DIVE", title_style))
    story.append(Paragraph("A Technical Guide to Multi-Modal AI Clinical Diagnostics & Explainable AI", subtitle_style))
    story.append(Spacer(1, 20))
    
    # ----------------------------------------------------
    # SECTION 1: EXECUTIVE OVERVIEW
    # ----------------------------------------------------
    story.append(Paragraph("1. Executive Overview & Clinical Rationale", h1_style))
    story.append(Paragraph(
        "In modern clinical settings, medical practitioners analyze diverse imaging modalities, "
        "including MRI and X-ray scans, to diagnose various pathologies. However, navigating across isolated "
        "viewing systems introduces diagnostic latency. MedVision AI addresses this by providing a unified "
        "dashboard linking multiple specialized deep learning classification models.", body_style
    ))
    story.append(Paragraph(
        "To ensure clinical adoption, the system integrates two fundamental pillars:", body_style
    ))
    story.append(Paragraph("- Explainable AI (XAI): Generates visual heatmaps (Grad-CAM) to overlay directly onto medical scans, pointing to the exact locations that influenced the model's classification decision.", bullet_style))
    story.append(Paragraph("- Automated Documentation: Uses report generation libraries to translate model classifications into clinical reports, exporting them instantly to a PDF format to reduce radiologist overhead.", bullet_style))
    story.append(Spacer(1, 10))

    # ----------------------------------------------------
    # SECTION 2: SYSTEM ARCHITECTURE
    # ----------------------------------------------------
    story.append(Paragraph("2. System Architecture & Component Communication", h1_style))
    story.append(Paragraph(
        "The system operates as a microservices architecture. A central dashboard hub (MedVision_Hub) "
        "coordinates client access to three distinct diagnostic services running on separate ports:", body_style
    ))
    story.append(Paragraph("- MedVision_Hub (Port 7865): Built with Gradio. Directs the user interface to launch the individual tools.", bullet_style))
    story.append(Paragraph("- BrainScan Insight (Port 7860): Built with Gradio. Runs an EfficientNet-B0 model trained to classify brain tumors.", bullet_style))
    story.append(Paragraph("- BoneExpert AI (Port 7862): Built with Gradio. Runs an EfficientNet-B3 model targeting bone fracture detection.", bullet_style))
    story.append(Paragraph("- ChestGuard AI (Port 8501): Built with Streamlit. Runs a DenseNet-121 model targeting pneumonia diagnosis.", bullet_style))
    
    story.append(Paragraph(
        "Selecting Gradio for Brain and Bone modules allows for interactive single-image input-output "
        "predictions. Choosing Streamlit for the Chest disease module provides a broad dashboard layout "
        "capable of rendering clean side-by-side original scans and heatmaps alongside multiple download operations.", body_style
    ))
    story.append(PageBreak())

    # ----------------------------------------------------
    # SECTION 3: DEEP DIVE INTO THE DIAGNOSTIC MODELS
    # ----------------------------------------------------
    story.append(Paragraph("3. Deep Dive into the Diagnostic Models", h1_style))
    
    # Brain MRI
    story.append(Paragraph("3.1 Brain MRI Tumor Detection (EfficientNet-B0)", h2_style))
    story.append(Paragraph(
        "Architecture Choice: EfficientNet-B0 scales network depth, width, and input resolution compoundly. "
        "With roughly 5.3 million parameters, it is lightweight, fast, and achieves high validation accuracy. "
        "This makes it highly suitable for resource-constrained server instances.", body_style
    ))
    story.append(Paragraph("Input preprocessing steps:", body_style))
    story.append(Paragraph("- Image resizing to 224x224 pixels to match the input dimensions of the model.", bullet_style))
    story.append(Paragraph("- Min-max scaling and division by 255.0 to scale pixels to [0, 1].", bullet_style))
    story.append(Paragraph("- Standardization using ImageNet channels: mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].", bullet_style))
    story.append(Paragraph("Explainable AI: Implemented using the pytorch_grad_cam library, targeting the features[-1] convolutional block to capture high-level semantic localization of tumors (Glioma, Meningioma, Pituitary).", body_style))
    
    # Bone Fracture
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.2 Bone Fracture Detection (EfficientNet-B3)", h2_style))
    story.append(Paragraph(
        "Architecture Choice: Bone fractures can be hairline thin and demand higher spatial resolution. "
        "EfficientNet-B3 takes an input size of 300x300 pixels, providing a 78% increase in pixels compared "
        "to B0. This captures finer bone texture details.", body_style
    ))
    story.append(Paragraph(
        "XAI Innovation (Custom Grad-CAM): To demonstrate deep understanding of PyTorch backpropagation "
        "without relying on external libraries, this module implements Grad-CAM natively using hooks:", body_style
    ))
    story.append(Paragraph("- Forward Hook: Registered on model.conv_head to store the forward feature map activations during inference.", bullet_style))
    story.append(Paragraph("- Backward Hook: Registered on model.conv_head to capture the gradients of the output score with respect to those activations.", bullet_style))
    story.append(Paragraph("- Mathematical combination: Computes weights by taking the mean of the gradients across spatial dimensions (global average pooling), multiplying activations by these weights, applying a ReLU activation, and normalizing the final output.", bullet_style))
    
    # Chest Disease
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.3 Chest X-Ray Pneumonia Detection (DenseNet-121)", h2_style))
    story.append(Paragraph(
        "Architecture Choice: DenseNet-121 connects every layer to every subsequent layer. In medical imaging, "
        "low-level structural details (like edges and lung field boundaries) must be preserved alongside "
        "high-level semantic patterns (like diffuse consolidation indicating pneumonia). Feature-reuse in "
        "DenseNet prevents the vanishing gradient problem and allows models to perform exceptionally well on small medical datasets.", body_style
    ))
    story.append(Paragraph(
        "Integration: Built within a Streamlit UI, providing dual file streaming handles. This allows users to download "
        "both the raw Grad-CAM JPG visualization and the structured ReportLab PDF diagnosis side-by-side.", body_style
    ))
    story.append(PageBreak())

    # ----------------------------------------------------
    # SECTION 4: INTERVIEW QUESTION BANK
    # ----------------------------------------------------
    story.append(Paragraph("4. Technical Interview Preparation & Questions", h1_style))
    story.append(Paragraph(
        "This section prepares you to confidently defend the architectural decisions, coding patterns, "
        "and deep learning fundamentals behind MedVision AI.", body_style
    ))
    
    # Q1
    story.append(Paragraph("Q1: What is the mathematical concept behind Grad-CAM?", q_style))
    story.append(Paragraph(
        "A: Grad-CAM calculates the gradient of the score for class c (before softmax) with respect to "
        "the feature map activations A of the last convolutional layer. These gradients are globally pooled "
        "to calculate a weight alpha representing the importance of each feature map. A weighted sum of the "
        "feature maps is then computed, followed by a ReLU operation (to focus only on features that positively "
        "correlate with the class), and upsampled to match the original image size.", a_style
    ))
    
    # Q2
    story.append(Paragraph("Q2: Why did you use three different architectures instead of training one model for all tasks?", q_style))
    story.append(Paragraph(
        "A: Each diagnostic domain has different visual characteristics. Brain tumor MRIs require multi-class classification "
        "where spatial position matters (EfficientNet-B0 balances efficiency and localization). Bone X-rays require "
        "detecting fine, structural lines (EfficientNet-B3 provides higher input resolution). Chest X-rays require "
        "identifying diffuse, hazy patterns of fluid or pneumonia (DenseNet-121's dense connections excel at feature reuse "
        "and gradient flow for diffuse patterns). Tailoring the architecture to the imaging modality optimizes performance.", a_style
    ))
    
    # Q3
    story.append(Paragraph("Q3: Explain the role of the hook mechanism in your custom Grad-CAM implementation.", q_style))
    story.append(Paragraph(
        "A: PyTorch discards intermediate gradients and activations during standard backpropagation to optimize memory. "
        "By registering a forward hook, we instruct the model to execute a callback function that copies the output activations "
        "of the target convolutional layer to a variable. By registering a backward hook, we capture the gradient of the loss "
        "with respect to those specific activations when backward() is called, allowing us to perform the Grad-CAM calculation.", a_style
    ))
    
    # Q4
    story.append(Paragraph("Q4: Why standardise inputs with ImageNet mean and standard deviation?", q_style))
    story.append(Paragraph(
        "A: The pre-trained weights (from EfficientNet and DenseNet) were trained on the ImageNet dataset. Standardizing "
        "our input scans using the same mean and standard deviation ensures our input distributions align with what the "
        "network expects, preventing covariate shift and ensuring the pre-trained features transfer effectively.", a_style
    ))
    
    # Q5
    story.append(Paragraph("Q5: How would you scale this prototype into a production-grade hospital system?", q_style))
    story.append(Paragraph(
        "A: To move to production: (1) Wrap the models in high-performance APIs using FastAPI or Triton Inference Server. "
        "(2) Containerize each service with Docker and deploy them to a scalable cloud platform using Kubernetes. "
        "(3) Implement DICOM (Digital Imaging and Communications in Medicine) protocol support to integrate directly with "
        "hospital PACS (Picture Archiving and Communication Systems). (4) Establish human-in-the-loop validation, where AI "
        "findings serve as pre-reads to be approved by a certified radiologist.", a_style
    ))

    # ----------------------------------------------------
    # BUILD DOCUMENT
    # ----------------------------------------------------
    doc.build(story)
    print("Project Guide PDF generated successfully.")

if __name__ == "__main__":
    create_project_guide()
