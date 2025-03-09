@ECHO off

SET VENV_DIR=venv_win_tflite
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

if not exist .\%PACKAGES_DIR%\pycoral-2.0.0-cp39-cp39-win_amd64.whl (
	curl.exe -L -o .\%PACKAGES_DIR%\pycoral-2.0.0-cp39-cp39-win_amd64.whl --url https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-win_amd64.whl
)

if not exist .\%PACKAGES_DIR%\tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl (
	curl.exe -L -o .\%PACKAGES_DIR%\tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl --url  https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl
)

if not exist .\%VENV_DIR%\Lib\site-packages\tensorflow\ (
	rem pip install tensorflow[and-cuda]
	pip install tensorflow
)

if not exist .\onnx2tflite (
	git clone https://github.com/MPolaris/onnx2tflite.git
	
	pushd .
	cd onnx2tflite
	python setup.py install
	popd
)

if not exist .\%VENV_DIR%\Lib\site-packages\pycoral\ (
	pip install .\%PACKAGES_DIR%\tflite_runtime-2.5.0.post1-cp39-cp39-win_amd64.whl
	pip install .\%PACKAGES_DIR%\pycoral-2.0.0-cp39-cp39-win_amd64.whl
	pip uninstall numpy -y && pip install numpy==1.26.4
)


if not exist .\%VENV_DIR%\Lib\site-packages\cv2\ (
	pip install opencv-python
)



pushd .
cd onnx2tflite
python -m onnx2tflite --weights "../../model/trained_model.onnx" --formats "tflite" --int8
python -m onnx2tflite --weights "../../model/trained_model.onnx" --formats "tflite"
popd