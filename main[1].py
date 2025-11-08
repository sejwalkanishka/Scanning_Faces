from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.core.detector import FaceDetector
from app.core.recognizer import FaceRecognizer

app = FastAPI(title='Face Recognition Service - InsightFace Based')

detector = FaceDetector()
recognizer = FaceRecognizer()

class DetectResponse(BaseModel):
    bbox: list
    score: float
    landmarks: list = None

@app.post('/detect')
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    try:
        results = detector.detect_bytes(data)
        return JSONResponse(content={'detections': results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/recognize')
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    try:
        results = detector.detect_bytes(data)
        out = []
        for d in results:
            emb = recognizer.get_embedding_from_crop(d['crop'])
            matches = recognizer.match(emb)
            out.append({'bbox': d['bbox'], 'score': d['score'], 'matches': matches})
        return {'results': out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/add_identity')
async def add_identity(name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    try:
        results = detector.detect_bytes(data)
        if not results:
            raise HTTPException(status_code=400, detail='No face detected')
        d = max(results, key=lambda x: x['score'])
        emb = recognizer.get_embedding_from_crop(d['crop'])
        recognizer.add_identity(name, emb)
        return {'status':'ok', 'name': name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/list_identities')
async def list_identities():
    return {'identities': recognizer.list_identities()}
