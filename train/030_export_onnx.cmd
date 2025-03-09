@ECHO off

SET VENV_DIR=venv_win_train
SET SRC_DIR=src
SET MODEL_OUTDIR=..\model
SET TRAIN_DIR=train
SET IMAGE_WIDTH=896
SET IMAGE_HEIGHT=504

if not exist .\%MODEL_OUTDIR% (
	mkdir .\%MODEL_OUTDIR%
)

if not exist .\%TRAIN_DIR% (
	echo Start the training first
	exit /B
)
call .\%VENV_DIR%\scripts\activate.bat

if not exist .\%VENV_DIR%\Lib\site-packages\paddle2onnx\ (
	pip install paddle2onnx
)


set CUDA_VISIBLE_DEVICES=0
python .\%SRC_DIR%\export.py ^
       --config .\%SRC_DIR%\pp_liteseg_small.yml ^
       --model_path .\%TRAIN_DIR%\best_model\model.pdparams ^
	   --save_dir .\%TRAIN_DIR% ^
	   --input_shape 1 3 %IMAGE_HEIGHT% %IMAGE_WIDTH% ^
	   --output_op softmax
	   
paddle2onnx --model_dir .\%TRAIN_DIR% ^
            --model_filename model.pdmodel ^
            --params_filename model.pdiparams ^
            --save_file .\%MODEL_OUTDIR%\trained_model.onnx