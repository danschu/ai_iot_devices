@ECHO off

SET VENV_DIR=venv_win_hailo
SET SRC_DIR=src
SET HAILO_WHL=hailort-4.20.0-cp39-cp39-win_amd64.whl

if not exist .\%VENV_DIR% (
	echo Preparing virtual environment...
	py -3.9 -m venv %VENV_DIR%
	if not exist .\%VENV_DIR% (
		exit /B
	)
)
call .\%VENV_DIR%\scripts\activate.bat
if not exist .\%VENV_DIR%\Lib\site-packages\hailo_platform\ (
	pip install "%ProgramFiles%\HailoRT\python\%HAILO_WHL%"
)
if not exist .\%VENV_DIR%\Lib\site-packages\cv2\ (
	pip install opencv-python
)



python .\%SRC_DIR%\predict_hailoai.py