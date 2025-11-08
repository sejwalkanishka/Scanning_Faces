import numpy as np
import onnxruntime as ort
import cv2
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
from collections import OrderedDict

class FaceRecognizer:
    def __init__(self, onnx_path='models/arcface.onnx', metric='cosine', top_k=3, threshold=0.35):
        self.onnx_path = onnx_path
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        except Exception:
            self.sess = None
        self.metric = metric
        self.top_k = top_k
        self.threshold = threshold
        self.emb_db = OrderedDict()
        self._build_index()

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112))
        img = img.astype('float32')
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2,0,1))
        return np.expand_dims(img, 0).astype('float32')

    def get_embedding_from_crop(self, crop_img):
        if self.sess is None:
            v = np.random.rand(512).astype('float32')
            v = v / (np.linalg.norm(v)+1e-10)
            return v
        x = self.preprocess(crop_img)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        emb = out.flatten().astype('float32')
        emb = emb / (np.linalg.norm(emb)+1e-10)
        return emb

    def add_identity(self, name, emb):
        self.emb_db[name] = emb.astype('float32')
        self._build_index()

    def list_identities(self):
        return list(self.emb_db.keys())

    def _build_index(self):
        if len(self.emb_db)==0:
            self.index = None
            return
        embs = np.vstack(list(self.emb_db.values())).astype('float32')
        d = embs.shape[1]
        if _HAS_FAISS:
            if self.metric=='cosine':
                self.index = faiss.IndexFlatIP(d)
                faiss.normalize_L2(embs)
            else:
                self.index = faiss.IndexFlatL2(d)
            self.index.add(embs)
            self.names = list(self.emb_db.keys())
        else:
            self.index = None
            self.names = list(self.emb_db.keys())

    def match(self, query_emb):
        if self.index is None:
            # brute-force fallback
            if len(self.emb_db)==0:
                return []
            embs = np.vstack(list(self.emb_db.values()))
            if self.metric=='cosine':
                qn = query_emb / (np.linalg.norm(query_emb)+1e-10)
                embsn = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-10)
                sims = embsn @ qn
                idxs = np.argsort(-sims)[:self.top_k]
                res = []
                for i in idxs:
                    s = float(sims[i])
                    if s < self.threshold: continue
                    res.append({'name': self.names[i], 'score': s})
                return res
            else:
                dists = np.linalg.norm(embs - query_emb, axis=1)
                idxs = np.argsort(dists)[:self.top_k]
                res = []
                for i in idxs:
                    s = float(1.0/(1.0+dists[i]))
                    if s < self.threshold: continue
                    res.append({'name': self.names[i], 'score': s})
                return res
        else:
            q = query_emb.reshape(1,-1).astype('float32')
            if self.metric=='cosine':
                faiss.normalize_L2(q)
            D,I = self.index.search(q, self.top_k)
            res = []
            for score, idx in zip(D[0], I[0]):
                if idx<0: continue
                s = float(score) if self.metric=='cosine' else float(1.0/(1.0+score))
                if s < self.threshold: continue
                res.append({'name': self.names[idx], 'score': s})
            return res
