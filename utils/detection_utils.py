from PIL import Image
from torchvision import transforms
import cv2
import constants

def preprocess_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Resize image to (160, 160) as FaceNet expects input shape (160, 160, 3)
    img = img.resize((160, 160), Image.LANCZOS)
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = transform(img).unsqueeze(0).to(constants.device)
    return img


def calculate_embedding(face,model):
    # Preprocess face image
    face = preprocess_image(face)
    # Generate embedding using FaceNet model
    embedding = model(face)
    return embedding