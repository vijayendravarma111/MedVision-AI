def generate_report(
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

        findings = """
Radiographic examination demonstrates cortical disruption
and abnormal bone alignment suggestive of a fracture.

Localized structural abnormality is identified within
the examined bony region.
"""

        impression = """
Bone fracture detected.

Radiographic findings are consistent with traumatic injury.
"""

        recommendation = """
- Routine clinical follow-up if symptoms persist.

- Additional imaging may be considered if clinically indicated.
"""

    # ========================================================
    # NORMAL
    # ========================================================

    else:

        findings = """
No obvious cortical disruption, fracture line,
or acute bony abnormality identified.

Bone alignment appears preserved.
"""

        impression = """
No radiographic evidence of fracture detected.
"""

        recommendation = """
• Routine clinical follow-up if symptoms persist.

• Additional imaging may be considered if clinically indicated.
"""

    report = {
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Findings": findings,
        "Impression": impression,
        "Recommendation": recommendation
    }

    return report