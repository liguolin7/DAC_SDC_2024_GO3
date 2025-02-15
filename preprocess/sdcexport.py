import cv2
import numpy as np
import torchvision
from models.experimental import attempt_load
import torch
import torch.nn as nn
import torchvision.ops
import onnx
import warnings
warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
def letterbox(im, new_shape=(512, 288), color=(114, 114, 114)):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_NEAREST)
    # Padding for left and top will be zero, so only add padding to right and bottom
    top, bottom = 0, dh
    left, right = 0, dw
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r

class BasicPreprocess(nn.Module):
    def __init__(self):
        super(BasicPreprocess, self).__init__()

    def forward(self, img):
        img = img.permute(0, 3, 1, 2)
        img = img.contiguous()
        img = img / 255.0
        return img

class ModelWithProcessing(nn.Module):
    def __init__(self, weights, device='cpu', inplace=True, fuse=True):
        super(ModelWithProcessing, self).__init__()
        self.preprocess = BasicPreprocess()
        self.model = attempt_load(weights, device, inplace, fuse)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        return x


weights = 'xx.pt'
model = ModelWithProcessing(weights, device='cpu', inplace=True, fuse=True)

bgr_img = cv2.imread(str('../dataset/images/train/01000.jpg'))

rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
img, _ = letterbox(rgb_img, (288, 512))  # padded resize
img = img[None]  # expand for batch dim
input_tensor = torch.from_numpy(img)
input_tensor = input_tensor.float()
onnx_name = "xx.onnx"

output2 = model(input_tensor)

model.eval()

torch.onnx.export(model, input_tensor, onnx_name, verbose=False, opset_version=12,
                  training=torch.onnx.TrainingMode.EVAL,
                  do_constant_folding=True,
                  input_names=['images'],
                  output_names=['output'],
                  dynamic_axes= None)

import onnxsim
model_onnx = onnx.load(onnx_name)
model_onnx, check = onnxsim.simplify(
    model_onnx,
    dynamic_input_shape=False
)

assert check, 'assert check failed'
onnx.save(model_onnx, onnx_name)
onnx_model = onnx.load(onnx_name)
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
except onnx.checker.ValidationError as e:
    print("ONNX model is invalid: ", e)
