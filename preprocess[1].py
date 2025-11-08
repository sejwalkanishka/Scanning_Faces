import cv2
import numpy as np

REF_POINTS = np.array([[30.2946, 51.6963],[65.5318,51.5014],[48.0252,71.7366],[33.5493,92.3655],[62.7299,92.2041]], dtype=np.float32)

def align_face(img, landmarks, output_size=(112,112)):
    try:
        src = np.array(landmarks, dtype=np.float32)
        dst = REF_POINTS
        tform = cv2.estimateAffinePartial2D(src, dst)[0]
        aligned = cv2.warpAffine(img, tform, output_size)
        return aligned
    except Exception:
        h,w = img.shape[:2]
        s = min(h,w)
        return cv2.resize(img[:s,:s], output_size)
