import os
import onnx

model_file = "./model/input_model.onnx"

onnx_model = onnx.load(model_file)
onnx_inputs = [x.name for x in onnx_model.graph.input]
onnx_outputs = [x.name for x in onnx_model.graph.output]

dims = onnx_model.graph.input[0].type.tensor_type.shape
shape = [1, dims.dim[1].dim_value, dims.dim[2].dim_value, dims.dim[3].dim_value] # 1, 3, 720, 1280

from hailo_sdk_client.runner.client_runner  import ClientRunner
outname, ext = os.path.splitext(model_file)
outname += ".har"

runner = ClientRunner(hw_arch="hailo8l")
 
hailo_model = runner.translate_onnx_model(
    model = model_file,
    start_node_names = onnx_inputs,
    end_node_names = onnx_outputs,
    net_input_shapes=[shape])

runner.save_har(outname)