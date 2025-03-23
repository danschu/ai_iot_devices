import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, quant_pre_process
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod 
from onnx import version_converter

import glob
import os
import cv2
import numpy as np

def prepare_input(image, input_width, input_height):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_width, input_height))
    
    input_img = input_img / 255.0 * 2 -1
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor

class quant_dataset_reader(CalibrationDataReader):
    def __init__(self, shape, max_files = 128):
        self.enum_data = None
        unconcatenated_batch_data = []
        
        dim, chan, input_height, input_width = shape
        
        for f in glob.glob("../train/dataset/bdd100k/images/100k/test/*.jpg"):
            img = cv2.imread(f)
            nchw_data = prepare_input(img, input_width, input_height)
            unconcatenated_batch_data.append(nchw_data)
            if len(unconcatenated_batch_data) >= max_files:
                break
            
        batch_data = np.concatenate(
            np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
        )
        self.nhwc_data_list = batch_data
        
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{"x": nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        

def main():
    basedir = "../model"
    
    model_input = os.path.join(basedir, 'trained_model_opt.onnx')
    model_output = os.path.join(basedir, 'trained_model_int8.onnx')

    # Load the model
    model = onnx.load(model_input)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    shape = None
    for _input in model.graph.input:
        if _input.name == "x":
            shape = [x.dim_value for x in _input.type.tensor_type.shape.dim]
    if shape is None:
        raise Exception("input 'x' not found in model")
    
    
    dataset_reader = quant_dataset_reader(shape)
   
    quantize_static(model_input, model_output, dataset_reader,
      #weight_type=QuantType.QInt8,
      #activation_type=QuantType.QInt8,
      #quant_format=QuantFormat.QDQ,
      #op_types_to_quantize = ["Conv"], 
      #nodes_to_quantize = None,
      #nodes_to_exclude = ["p2o.Conv.0", "p2o.Conv.1", "p2o.Conv.2", "p2o.Conv.3", "p2o.Conv.4", "p2o.Conv.5"]
    )


if __name__ == "__main__":
    main()