from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

MODEL_NAME = "giacomoarienti/nsfw-classifier"

device = torch.device("cpu")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def scan_image(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    id2label = model.config.id2label
    scores = {id2label[i]: float(probs[i]) for i in range(len(id2label))}

    primary = max(scores, key=scores.get)
    safe = primary not in ["hentai", "porn", "sexy"]

    return primary, scores, safe

