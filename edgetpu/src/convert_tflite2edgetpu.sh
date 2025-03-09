#!/bin/bash

edgetpu_compiler -s ../model/trained_model_int8.tflite -o ../model/ -a