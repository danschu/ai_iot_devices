import os
import glob
import tensorflow as tf
import numpy as np
import cv2
import time
import pathlib

input_height, input_width = 768, 1280
model_path = "../model/trained_model.tflite"
model_path_int8 = "../model/trained_model_int8.tflite"
dataset_dir = "../train/dataset/bdd100k/images/100k/test/"

def prepare_input(image, input_width, input_height):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_width, input_height))
    input_img = input_img / 255.0
    #input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor

global idx
idx = 0
def representative_data_gen():
    for image_file in glob.glob(f"{dataset_dir}/*.jpg"):
        global idx
        if idx == 32:
            break
        print(image_file)
        idx += 1
        img = cv2.imread(image_file)
        image = prepare_input(img, input_width, input_height)
        
        #image = np.random.rand(1, input_height, input_width, 3).astype(np.float32)
    
        data = np.array([[1.0, 1.0]]).astype(np.float32)
        yield [image, data]


model = tf.saved_model.load(model_path)
converter = tf.lite.TFLiteConverter.from_concrete_functions([model])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.inference_type = tf.int8
#converter.inference_input_type = tf.float32
#converter.inference_output_type = tf.float32
#converter.inference_type = tf.int16

quant_model = converter.convert()
tflite_model_file = pathlib.Path(model_path_int8)
tflite_model_file.write_bytes(quant_model)

"""
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
    
def convert_output(data, output_details):
    output_type = output_details['dtype']
    if output_type == np.int8 or output_type == np.uint8:
        output_scale, output_zero_point = output_details['quantization']
        print("Ouput scale:", output_scale)
        print("Ouput zero point:", output_zero_point)
        print()
        data = data.astype(np.float32)
        data = (data-output_zero_point)*output_scale
    return data



interpreter = tf.lite.Interpreter(model_path=model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

print(input_details)


for image_file in glob.glob("testimages/*.jpg"):
    img = cv2.imread(image_file)  
    t1 = time.time()
    image_height, image_width, _ = img.shape    

    im_small = prepare_input(img, input_width, input_height)      
    im_small = convert_input(im_small, input_details[0])

    data = np.array([[1.0, 1.0]]).astype(np.float32)   
    data = convert_input(data, input_details[1])
    

    interpreter.set_tensor(input_details[0]['index'], im_small)
    interpreter.set_tensor(input_details[1]['index'], data)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    output = convert_output(output, output_details[0])
    print(output)
    
    img_blur = cv2.blur(img, (37,37))
    
    for i in output:
        if i[1] > 0.2:
            #print(i)
            x1 = int(i[2]/input_width*image_width)
            y1 = int(i[3]/input_height*image_height)
            x2 = int(i[4]/input_width*image_width)
            y2 = int(i[5]/input_height*image_height)
            img[y1:y2,x1:x2] = img_blur[y1:y2,x1:x2]
        
            #img = cv2.rectangle(img, (x1, y1),
            #    (x2, y2), (255, 0, 0), -1)
            
    
    t2 = time.time()
    print("FPS:", 1/((t2-t1)+0.001))
    
    cv2.namedWindow("test")
    cv2.imshow("test", img)
    cv2.waitKey(10)
"""