@ECHO off
SET PACKAGES_DIR=packages

if not exist .\%PACKAGES_DIR%\edgetpu_runtime_20221024.zip (
	curl.exe -L -o .\%PACKAGES_DIR%\edgetpu_runtime_20221024.zip ^
		--url https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
)

if not exist .\%PACKAGES_DIR%\edgetpu_runtime\ (
	echo Extracting edge runtime...
	tar -xf .\%PACKAGES_DIR%\edgetpu_runtime_20221024.zip -C .\%PACKAGES_DIR%
	echo Edge TPU will now be installed, press any key...
	pause
	.\%PACKAGES_DIR%\edgetpu_runtime\install.bat
)