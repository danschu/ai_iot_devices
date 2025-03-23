import sys
sys.path.append("../train/src")

from predict_onnx import predict_onnx


if __name__ == "__main__":
    predict_onnx(
        modelfile = "../model/trained_model_int8.onnx",
        imagefile = "../train/dataset/bdd100k/images/100k/test/cb4bfc16-80e9d4a2.jpg"
    )