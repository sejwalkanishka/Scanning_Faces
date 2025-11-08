from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.detector import FaceDetector
from core.recognizer import FaceRecognizer

app = FastAPI()

# Initialize the Face Detector and Recognizer
detector = FaceDetector()
recognizer = FaceRecognizer()

class ImageData(BaseModel):
    image_path: str

@app.post("/detect/")
async def detect_faces(data: ImageData):
    try:
        faces = detector.detect(data.image_path)
        return {"faces": faces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize/")
async def recognize_face(data: ImageData):
    try:
        identity, confidence = recognizer.recognize(data.image_path)
        return {"identity": identity, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_identity/")
async def add_identity(data: ImageData):
    try:
        recognizer.add_identity(data.image_path)
        return {"message": "Identity added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_identities/")
async def list_identities():
    return {"identities": recognizer.list_identities()}
