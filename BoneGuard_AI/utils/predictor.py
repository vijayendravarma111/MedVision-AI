import torch
import timm

from PIL import Image
from torchvision import transforms

DEVICE = "cpu"

CLASS_NAMES = {
    0: "Fracture Detected",
    1: "Normal Bone"
}

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "boneguard_final.pth",
        map_location=DEVICE
    )
)

model.eval()

def predict_xray(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    prediction = CLASS_NAMES[
        pred.item()
    ]

    confidence = round(
        confidence.item() * 100,
        2
    )

    return (
        prediction,
        confidence
    )