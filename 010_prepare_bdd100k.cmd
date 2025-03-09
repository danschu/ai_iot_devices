@ECHO off

SET SRC_DIR=src
SET DATASET_DIR=dataset
SET VENV_DIR=venv_win_prepare

if not exist .\%DATASET_DIR% (
  mkdir .\%DATASET_DIR%
)

if not exist .\%DATASET_DIR%\bdd100k_images_100k.zip (
	curl.exe -o .\%DATASET_DIR%\bdd100k_images_100k.zip --url http://128.32.162.150/bdd100k/bdd100k_images_100k.zip
)

if not exist .\%DATASET_DIR%\bdd100k_labels.zip (
	curl.exe -o .\%DATASET_DIR%\bdd100k_labels.zip --url http://128.32.162.150/bdd100k/bdd100k_labels.zip
)


if not exist .\%DATASET_DIR%\bdd100k\ (
	echo Extracting images...
	tar -xf .\%DATASET_DIR%\bdd100k_images_100k.zip -C .\%DATASET_DIR%
)

if not exist .\%DATASET_DIR%\100k\ (
	echo Extracting labels...
	tar -xf .\%DATASET_DIR%\bdd100k_labels.zip -C .\%DATASET_DIR%
)

if not exist .\%VENV_DIR% (
	echo Preparing virtual environment...
	py -3.11 -m venv %VENV_DIR%
	if not exist .\%VENV_DIR% (
		exit /B
	)
)

call .\%VENV_DIR%\scripts\activate

if not exist .\%VENV_DIR%\Lib\site-packages\cv2\ (
	pip install opencv-python
)

if not exist .\%VENV_DIR%\Lib\site-packages\huggingface_hub\ (
	pip install huggingface_hub
)

if not exist .\%VENV_DIR%\Lib\site-packages\torch\ (
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)
if not exist .\%VENV_DIR%\Lib\site-packages\PIL\ (
	pip install pillow
)
if not exist .\%VENV_DIR%\Lib\site-packages\seaborn\ (
	pip install seaborn
)
if not exist .\%VENV_DIR%\Lib\site-packages\sam2\ (
	pip install git+https://github.com/facebookresearch/sam2.git
)


python .\%SRC_DIR%\prepare_bdd100k.py
