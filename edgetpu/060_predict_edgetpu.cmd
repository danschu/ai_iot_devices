@ECHO off

SET VENV_DIR=venv_win_edgetpu
SET SRC_DIR=src
SET PACKAGES_DIR=packages

if not exist .\%PACKAGES_DIR% (
  mkdir .\%PACKAGES_DIR%
)

if not exist .\%VENV_DIR% (
	echo Preparing virtual environment...
	
	py -3.9 -m venv %VENV_DIR%
	if not exist .\%VENV_DIR% (
		exit /B
	)
)


call .\%VENV_DIR%\scripts\activate
if not exist .\%VENV_DIR%\Lib\site-packages\pycoral\ (
	pip install numpy==1.26.4
	pip install .\%PACKAGES_DIR%\tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl
	pip install .\%PACKAGES_DIR%\pycoral-2.0.0-cp39-cp39-win_amd64.whl
)
if not exist .\%VENV_DIR%\Lib\site-packages\cv2\ (
	pip install opencv-python
)

python .\%SRC_DIR%\predict_edgetpu.py