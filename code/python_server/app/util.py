# util.py
# Utility functions

# Library
import os
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from app.config import DEVICE, DETECTOR_PATH, CLASSIFIER_WEIGHT_PATH, EMOTION, EMOTION_MAP, IMAGE_SIZE
from model.model_arch import EmotionClassifier

# Load detector
def load_detector():
	model = YOLO(DETECTOR_PATH, task="detect")
	return model

# Load classifier
def load_classifier():
	model = EmotionClassifier()
	model.load_state_dict(torch.load(CLASSIFIER_WEIGHT_PATH, map_location=DEVICE))
	model.eval()
	return model

# Crop image
def crop_face(model, image):
	result = model(image)[0]
	boxes = result.boxes

	for i, box in enumerate(boxes):
		x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
		image = image.crop((x_min, y_min, x_max, y_max))
	
	return image

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Predict expression from image
def predict_expression(model, image):
	image_tensor = transform(image).unsqueeze(0).to(DEVICE)

	with torch.inference_mode():
		outputs = model(image_tensor)
		confidences = torch.softmax(outputs, dim=1)
		
		if torch.any(confidences >= 0.5): # Check if the emotion is distinct
			idx = torch.argmax(confidences).item()
			emotion = EMOTION[idx]
		else:
			emotion = "blank"
	
	return EMOTION_MAP[emotion]