@ECHO off
SET UBUNTU_DIR=Ubuntu_edgetpu
SET SRC_DIR=src


IF not exist .\%UBUNTU_DIR% (
	echo Please install %UBUNTU_DIR% in WSL first - run 040_install_ubuntu_coral
	pause
	exit /B
)

wsl.exe -d %UBUNTU_DIR% < .\%SRC_DIR%\convert_tflite2edgetpu.sh