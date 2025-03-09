# Image Anonymization on IoT devices on windows

## 1) Train and test a segmentation model which detects cars and pedestrians

### Prepare [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)-Dataset 

Run the command

```cmd
010_prepare_bdd100k.cmd
```

It first downloads the dataset and extracts it to ./dataset/. Then it uses [SAM2](https://github.com/facebookresearch/sam2) to convert the bounding-boxes to segmentations masks (BDD100k has segmentation masks / polygons but only for 10k images). In this example we are only using 1000 images for teh training and 100 images for the validation. You can chnge this in the [prepare_bddk100k.py](./src/prepare_bdd100k.py) file.


### Train the datset using [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/README_EN.md)

Run the command

```cmd
020_train_paddleseg.cmd
```

It used PaddleSeg with the [ppliteseg-config](./src/pp_liteseg.yml) to train the model.

```
Note1: PaddleSeg uses -1..1 normalization, but it is changed in to 0..1 [pytorch equivalent] by using mean[0,0,0] in the config-file.
```

### Export the model to ONNX

Run the command

```cmd
030_export_onnx.cmd
```

It export the model to onnx.
```
Note1: If you changed the config you also need to change the export-resolution (currently 1280x1024 is used in training and export).
Note2: SoftMax is used instead of ArgMax. 
```

### Test the ONNX-model

Run the command

```cmd
040_predict_onnx.cmd
```

This uses the model and tests it on one image using CPU- and DirectML-Execution-Providers.

## 2) Convert and Test the model with HailoRT

Create an account at https://hailo.ai and go to download portal https://hailo.ai/developer-zone/software-downloads/

### Download HailoRT for Windows and put it in sub-directory 'packages'
* Software Package: AI Software Suite
* Software Sub-Package: HailoRT
* Architecture x86
* OS: Windows
  * @ HailoRT – Windows installer
    * hailort_4.20.0_windows_installer.msi

#### Install it by running the msi-file
```cmd
./packages/hailort_4.20.0_windows_installer.msi
```

#### Test installation with commandline
```cmd
hailortcli scan
```

#### Output should be something like:
```
Hailo Devices:
[-] Device: 0000:04:00.0
```

### Download Dataflow Compiler for Linux (WSL) and put it in sub-directory 'packages'
* Software Package: AI Software Suite
* Software Sub-Package: Dataflow Compiler
* Architecture x86
* OS: Linux
  * @ Hailo Dataflow Compiler – Python package (whl)
    * hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl


### Install required modules for the model conversion (Ubuntu @ WSL). 

Open commandline and install a copy of Ubuntu-20.04 and the HailAI requirements by running

```cmd
110_install_ubuntu_hailoai.cmd
```

### Convert the model from ONNX to Hailo fileformat (har)

Run the command

```cmd
120_convert_onnx2har.cmd
```

This covnerts the model from onnx to har in multiple steps (it will take ~1h).


### Test the converted model

Run the command

```cmd
130_predict_hailoai.cmd
```

