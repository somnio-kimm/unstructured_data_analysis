from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import base64
from PIL import Image
import io

app = FastAPI()

# Open webcam once globally
camera = cv2.VideoCapture(0)

@app.get("/")
def root():
	return {"message": "Python Server is running"}

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


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
	try:
		contents = await file.read()
		image = Image.open(io.BytesIO(contents)).convert("RGB")

		# 🔮 Replace this with your real model prediction
		emotion = "happy"  # dummy

		return JSONResponse(content={"emotion": emotion})
	except Exception as e:
		return JSONResponse(content={"error": str(e)}, status_code=500)