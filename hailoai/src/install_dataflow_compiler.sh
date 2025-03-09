#!/bin/bash


DATAFLOW_COMPILER_FILE=hailo_dataflow_compiler-3.30.0-py3-none-linux_x86_64.whl
PACKAGES_SUB_DIR=packages

if [ ! -d $PACKAGES_SUB_DIR ]; then
	mkdir $PACKAGES_SUB_DIR
fi

if [ ! -f ./$PACKAGES_SUB_DIR/$DATAFLOW_COMPILER_FILE ]; then
  if [ ! -f $DATAFLOW_COMPILER_FILE ]; then
    echo "Please download the dataflow compiler first and put it in sub-directory '"$PACKAGES_SUB_DIR"'"
	echo "https://hailo.ai/developer-zone/software-downloads/"
	exit 1
  else
    mv $DATAFLOW_COMPILER_FILE ./$PACKAGES_SUB_DIR/$DATAFLOW_COMPILER_FILE
  fi
fi

sudo apt-get update
sudo apt-get install libpython3-dev python3-venv python3-dev python3-pip graphviz libgraphviz-dev gcc g++ python3-tk -y
python3 -m venv venv_linux_hailo
source venv_linux_hailo/bin/activate
pip3 install pandas
pip3 install ./$PACKAGES_SUB_DIR/$DATAFLOW_COMPILER_FILE