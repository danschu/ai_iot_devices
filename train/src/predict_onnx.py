import cv2 # pip install opencv-python
import numpy as np 
import onnxruntime # pip install onnxruntime
import os
import time

modelfile = "../model/trained_model.onnx"

for prov in ["CPUExecutionProvider", "DmlExecutionProvider"]: 
    session = onnxruntime.InferenceSession(modelfile, providers=[prov])
            
    model_inputs = session.get_inputs()
    model_outputs = session.get_outputs()        

    input_shape = model_inputs[0].shape
            
    input_width = input_shape[3]
    input_height = input_shape[2]
        
    print(input_width, input_height) 

    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    img_orig = cv2.imread("./dataset/bdd100k/images/100k/test/cb4bfc16-80e9d4a2.jpg")

    cnt = 20
    print(f"Running {cnt} times...")
    t1 = time.time()
    for idx in range(cnt):
        input_img = cv2.resize(img_orig, (input_width, input_height))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)    
        input_img = np.asarray(input_img, np.float32)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[None, :, :, :]
        outputs = session.run(output_names, {input_names[0]: input_tensor})
        detection_mask = np.argmax(outputs[0][0], 0)
        detection_mask = np.array(detection_mask).astype(np.uint8)
        detection_mask *= 80
        
    detection_mask = np.dstack([detection_mask, detection_mask, detection_mask])            
    image_blend = cv2.resize(img_orig, (input_width, input_height))
    image_blend[detection_mask > 0] = 0
            
    t2 = time.time()
    print(f"Frames per second ({prov}): {(cnt)/(0.0001+t2-t1)}")

cv2.namedWindow("test")
cv2.imshow("test", image_blend)
cv2.waitKey(0)