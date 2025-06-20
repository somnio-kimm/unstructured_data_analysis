# util.py
# Load model & predict

# Library
import torch
from torchvision import transforms
from PIL import Image
from app.config import DEVICE, MODEL_WEIGHT_PATH, EMOTION, EMOTION_MAP, IMAGE_SIZE
from model.model_arch import EmotionClassifier

# Load model
def load_model():
	model = EmotionClassifier()
	model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
	model.eval()
	return model

transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# Predict expression from image
def predict_expression(model, image_path):
	image = Image.open(image_path).convert('RGB')
	image_tensor = transform(image).unsqueeze(0).to(DEVICE)

	with torch.inference_mode():
		outputs = model(image_tensor)
		outputs = torch.softmax(outputs, dim=1)
		print("Softmax scores:", outputs)
		print("Predicted index:", outputs.argmax().item())
	idx = outputs.argmax(dim=1).item()
	emotion = EMOTION[idx]
	return emotion
	#return EMOTION_MAP[emotion]