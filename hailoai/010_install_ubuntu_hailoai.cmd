@ECHO off
SET UBUNTU_DIR=Ubuntu_HailoAI
SET SRC_DIR=src
SET DATAFLOW_COMPILER_FILE=hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl
SET PACKAGES_DIR=packages


IF not exist .\%PACKAGES_DIR%/ (
	mkdir .\%PACKAGES_DIR%/
)

IF not exist .\%PACKAGES_DIR%\%DATAFLOW_COMPILER_FILE% (
	IF exist %DATAFLOW_COMPILER_FILE% (
		move %DATAFLOW_COMPILER_FILE% .\%PACKAGES_DIR%\%DATAFLOW_COMPILER_FILE%
	) else (
		echo Please download the dataflow compiler first and put it in sub-directory '%PACKAGES_DIR%'
		echo https://hailo.ai/developer-zone/software-downloads/
		pause
		exit /B
	)
)

IF not exist %UBUNTU_DIR% (
	mkdir %UBUNTU_DIR%
	wsl.exe --unregister %UBUNTU_DIR%
	wsl.exe --install --d Ubuntu-22.04 < .\%SRC_DIR%\logout.txt
	wsl.exe --export Ubuntu-22.04 .\%UBUNTU_DIR%\ubuntu-22.04.tar
	wsl.exe --import %UBUNTU_DIR% .\%UBUNTU_DIR%\%UBUNTU_DIR% .\%UBUNTU_DIR%\ubuntu-22.04.tar
)

wsl.exe -d %UBUNTU_DIR% < .\%SRC_DIR%\install_dataflow_compiler.sh