import time
import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
import cv2


# Loading compiled HEFs to device:
hef_path = "model.hef"
hef = HEF(hef_path)
    
# The target can be used as a context manager ("with" statement) to ensure it's released on time.
# Here it's avoided for the sake of simplicity
params = VDevice.create_params()

with VDevice(params) as target:

    # Configure network groups
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()
    
    # Create input and output virtual streams params
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)    
    
    # Define dataset params
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info.shape
    print(input_vstream_info.shape)
    print(output_vstream_info.shape)

    
    img_orig = cv2.imread("./dataset/bdd100k/images/100k/test/cb4bfc16-80e9d4a2.jpg")

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            t1 = time.time()
            cnt = 20
            print(f"Running {cnt} times...")
            for i in range(cnt):
                image_small = cv2.resize(img_orig, (image_width, image_height)).astype(np.float32)
                image_prepared = image_small / 255.0
                image_prepared = np.expand_dims(image_prepared, 0)
                input_data = {input_vstream_info.name: image_prepared}
    
                infer_results = infer_pipeline.infer(input_data)
                #print('Stream output shape is {}'.format(infer_results[output_vstream_info.name].shape))
                detection_mask = infer_results[output_vstream_info.name][0]
                detection_mask = np.argmax(detection_mask, -1)
                #print(detection_mask.shape)
                detection_mask = (detection_mask*80).astype(np.uint8)
            t2 = time.time()
            print(f"frames per second {(cnt)/(0.0001+t2-t1)}")
            
            detection_mask = np.dstack([detection_mask, detection_mask, detection_mask])            
            image_blend = cv2.resize(img_orig, (image_width, image_height))
            image_blend[detection_mask > 0] = 0
            
            cv2.namedWindow("test")
            cv2.imshow("test", image_blend)
            cv2.waitKey(0)
   