source venv_linux_hailo/bin/activate
python3 src/convert_onnx2har.py
hailo optimize ./model/input_model.har --use-random-calib-set --hw-arch hailo8l
hailo compiler model_optimized.har