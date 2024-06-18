
import torch
from torchvision import models
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_to_onnx(model_path, onnx_path):
    model = models.resnet18(pretrained=False)
    num_classes = ...  # set the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])

def convert_onnx_to_tf(onnx_path, tf_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)

def convert_tf_to_tflite(tf_path, tflite_path):
    loaded_model = tf.saved_model.load(tf_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    model_path = 'best_model.pth'
    onnx_path = 'model.onnx'
    tf_path = 'model_tf'
    tflite_path = 'model.tflite'

    convert_to_onnx(model_path, onnx_path)
    convert_onnx_to_tf(onnx_path, tf_path)
    convert_tf_to_tflite(tf_path, tflite_path)