import numpy as np
import cv2
import io
try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception:
    _HAS_INSIGHT = False

class FaceDetector:
    def __init__(self, provider='cpu'):
        self.provider = provider
        if _HAS_INSIGHT:
            self.app = FaceAnalysis(providers=[provider])
            try:
                self.app.prepare(ctx_id=-1, det_size=(640,640))
            except Exception:
                self.app.prepare()
        else:
            self.app = None

    def _bgr_from_bytes(self, b):
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def detect(self, image_path):
        img = cv2.imread(image_path)
        return self._detect_img(img)

    def detect_bytes(self, b):
        img = self._bgr_from_bytes(b)
        return self._detect_img(img)

    def _detect_img(self, img):
        h,w = img.shape[:2]
        if self.app is None:
            crop = cv2.resize(img, (112,112))
            return [{'bbox':[0,0,w,h], 'score':0.9, 'landmarks':None, 'crop':crop}]
        faces = self.app.get(img)
        out = []
        for f in faces:
            x1,y1,x2,y2 = [int(x) for x in f.bbox]
            score = float(getattr(f, 'det_score', getattr(f,'score',0.0)))
            crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            out.append({'bbox':[x1,y1,x2,y2], 'score':score, 'landmarks': getattr(f,'kps',None).tolist() if getattr(f,'kps',None) is not None else None, 'crop':crop})
        return out
