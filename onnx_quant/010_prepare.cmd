@ECHO off

SET SRC_DIR=src
SET MODEL_DIR=..\model
SET VENV_DIR=venv_win_onnx

if not exist .\%VENV_DIR% (
	echo Preparing virtual environment...
	py -3.11 -m venv %VENV_DIR%
	if not exist .\%VENV_DIR% (
		exit /B
	)
)

call .\%VENV_DIR%\scripts\activate

if not exist .\%VENV_DIR%\Lib\site-packages\onnxruntime\ (
	pip install onnxruntime
)
if not exist .\%VENV_DIR%\Lib\site-packages\onnx\ (
	pip install onnx
)

python -m onnxruntime.quantization.preprocess --input "%MODEL_DIR%\trained_model.onnx" --output "%MODEL_DIR%\trained_model_opt.onnx"
python %SRC_DIR%\static_quant.py
