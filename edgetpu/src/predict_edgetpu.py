import numpy as np
from PIL import Image
import cv2
import time
import pathlib


from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def prepare_input(image, input_width, input_height):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_width, input_height))
    input_img = input_img / 255.0
    #input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor
    
def convert_input(data, input_details):
    input_type = input_details['dtype']
    if input_type == np.int8 or input_type == np.uint8:
        input_scale, input_zero_point = input_details['quantization']
        print("Input scale:", input_scale)
        print("Input zero point:", input_zero_point)
        print()
        data = (data / input_scale) + input_zero_point
        data = np.around(data)
        data = data.astype(input_type)
    return data

def main(image_file, model_file):
    interpreter = make_interpreter(model_file, device=':0')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
  
    print(input_details)
    
    _, height_tensor, width_tensor, _ = input_details[0]["shape"]
        
        
    cnt = 20
    img_orig = cv2.imread(image_file)  
    print(f"Running {cnt} times...")
    t1 = time.time()
    for idx in range(cnt):
        img_prepared = prepare_input(img_orig, width_tensor, height_tensor)      
        img_prepared = convert_input(img_prepared, input_details[0])
        interpreter.set_tensor(input_details[0]['index'], img_prepared)
        interpreter.invoke()
        
        detection_mask = interpreter.get_tensor(output_details[0]['index'])
        #detection_mask = convert_output(detection_mask, output_details[0])
        detection_mask = np.argmax(detection_mask[0], 0)
        detection_mask = np.array(detection_mask).astype(np.uint8)
        detection_mask *= 80
        
    detection_mask = np.dstack([detection_mask, detection_mask, detection_mask])            
    image_blend = cv2.resize(img_orig, (width_tensor, height_tensor))
    image_blend[detection_mask > 0] = 0
            
    t2 = time.time()
    print(f"Frames per second: {(cnt)/(0.0001+t2-t1)}")

    cv2.namedWindow("test")
    cv2.imshow("test", image_blend)
    cv2.waitKey(0)

if __name__ == '__main__':
    image_file = "../train/dataset/bdd100k/images/100k/test/cb4bfc16-80e9d4a2.jpg"
    model_file = "../model/trained_model_int8_edgetpu.tflite"
    main(image_file, model_file)