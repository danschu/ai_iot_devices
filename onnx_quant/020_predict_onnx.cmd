@ECHO off

SET VENV_DIR=venv_win_onnx
SET SRC_DIR=src

if not exist .\%VENV_DIR% (
	echo Preparing virtual environment...
	py -3.11 -m venv %VENV_DIR%
	if not exist .\%VENV_DIR% (
		exit /B
	)
)

call .\%VENV_DIR%\scripts\activate.bat

if not exist .\%VENV_DIR%\Lib\site-packages\onnxruntime\ (
	pip install onnxruntime
)
if not exist .\%VENV_DIR%\Lib\site-packages\cv2\ (
	pip install -U opencv-python
)

python .\%SRC_DIR%\predict_onnx_int8.py