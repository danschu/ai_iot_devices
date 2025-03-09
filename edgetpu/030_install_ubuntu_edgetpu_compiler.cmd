@ECHO off
SET UBUNTU_DIR=Ubuntu_edgetpu
SET SRC_DIR=src
SET PACKAGES_DIR=packages


IF not exist %UBUNTU_DIR% (
	mkdir %UBUNTU_DIR%
	wsl.exe --unregister %UBUNTU_DIR%
	wsl.exe --install --d Ubuntu-22.04 < .\%SRC_DIR%\logout.txt
	wsl.exe --export Ubuntu-22.04 .\%UBUNTU_DIR%\ubuntu-22.04.tar
	wsl.exe --import %UBUNTU_DIR% .\%UBUNTU_DIR%\%UBUNTU_DIR% .\%UBUNTU_DIR%\ubuntu-22.04.tar
)

wsl.exe -d %UBUNTU_DIR% < .\%SRC_DIR%\install_edgetpu_compiler.sh