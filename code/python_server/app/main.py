# main.py
# Communicate with game engine

# Libraries
import base64
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

from app.util import load_detector, load_classifier, crop_face, predict_expression

# Global webcam instance
camera = None

# FastAPI app with camera lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
	global camera
	camera = cv2.VideoCapture(0)
	print("Camera started")
	yield
	camera.release()
	print(" Camera released")

# Instantiation
app = FastAPI(lifespan=lifespan)
detection_model = load_detector()
classification_model = load_classifier()

# Root route
@app.get("/")
def root():
	return {"message": "Facial Expression API is running."}

@app.get("/capture_image/")
def capture_image():
	if not camera.isOpened():
		return JSONResponse(content={"error": "Camera not opened"}, status_code=500)

	ret, frame = camera.read()
	if not ret:
		return JSONResponse(content={"error": "Failed to read from camera"}, status_code=500)

	# Resize for smaller size (helps with Godot too)
	frame = cv2.resize(frame, (244, 244))

	success, buffer = cv2.imencode('.jpg', frame)
	if not success:
		return JSONResponse(content={"error": "Failed to encode image"}, status_code=500)

	img_bytes = base64.b64encode(buffer).decode("utf-8")
	return JSONResponse(content={"image": img_bytes})

# Webcam-based prediction
@app.post("/predict/")
async def predict():
	ret, frame = camera.read()
	if not ret:
		return JSONResponse({"error": "Webcam capture failed"}, status_code=500)

	try:
		# Convert BGR (OpenCV) to RGB (PIL)
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)

		# Crop face and classify expression
		image_cropped = crop_face(model=detection_model, image=image)
		emotion = predict_expression(model=classification_model, image=image_cropped)

		return JSONResponse({"emotion": emotion})
	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)