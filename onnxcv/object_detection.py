import cv2 
import numpy as np
import onnxruntime as rt
from skimage.transform import resize

'''
This is the object detection model.
To use this, import the model as follows:

#########################################

from onnxcv import ObjectDetector
model = ObjectDetector('path/to/model.onnx')
model.run()

'''

class ObjectDetector:
    def __init__(self, model):
        self.model = model

    def run(self):
        while True:
            # Starting the camera
            cam = cv2.VideoCapture(0)
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resizing data
            X = np.array(img)
            X = resize(X, (3, 416, 416))
            X = X[np.newaxis, :, :, :]
            X = X.astype(np.float32)

            # Running the model
            sess = rt.InferenceSession(self.model)
            input_name = sess.get_inputs()[0].name
            pred_onnx = sess.run(None, {input_name: X})
            print(pred_onnx)

            # Showing the output
            cv2.imshow('Detect',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
