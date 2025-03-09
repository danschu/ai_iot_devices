@ECHO off

SET VENV_DIR=venv_win_train
SET SRC_DIR=src
SET DATASET_DIR=dataset

if not exist .\%VENV_DIR% (
	echo Start the training first
	exit /B
)
call .\%VENV_DIR%\scripts\activate.bat

if not exist .\%VENV_DIR%\Lib\site-packages\paddle2onnx\ (
	pip install paddle2onnx
)

set CUDA_VISIBLE_DEVICES=0
python .\%SRC_DIR%\export.py ^
       --config .\%SRC_DIR%\pp_liteseg.yml ^
       --model_path .\%DATASET_DIR%\train\best_model\model.pdparams ^
	   --save_dir .\%DATASET_DIR%\train ^
	   --input_shape 1 3 768 1280 ^
	   --output_op softmax
	   
paddle2onnx --model_dir .\%DATASET_DIR%\train ^
            --model_filename model.pdmodel ^
            --params_filename model.pdiparams ^
            --save_file .\model\input_model.onnx