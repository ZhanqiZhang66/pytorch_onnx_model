# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
#%%
# Super Resolution model definition in PyTorch
"""
This model uses the efficient sub-pixel convolution layer described in 
“Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel 
Convolutional Neural Network” - Shi et al for increasing the resolution of an image 
by an upscale factor. The model expects the Y component of the YCbCr of an image as 
an input, and outputs the upscaled Y component in super resolution.
"""
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)
#%% Ordinarily, you would now train this model;
#%% Here, Load pretrained model weights

model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1  # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
# set the model to inference mode
torch_model.eval()
'''
It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, 
to turn the model to inference mode. 
This is required since operators like dropout or batchnorm 
behave differently in inference and training mode.
'''
#%%

'''
The ONNX exporter can be both trace-based and script-based exporter.

Trace-based means that it operates by executing your model once, 
and exporting the operators which were actually run during this run. 
This means that if your model is dynamic, e.g., changes behavior depending on input data,
the export won’t be accurate.examining the model trace and making sure the traced operators look reasonable. 
If your model contains control flows like for loops and if conditions, 
trace-based exporter will unroll the loops and if conditions, 
exporting a static graph that is exactly the same as this run.
 
If you want to export your model with dynamic control flows, you will need to use the script-based exporter.

Script-based means that the model you are trying to export is a ScriptModule. ScriptModule is the core data structure in TorchScript
https://pytorch.org/docs/master/onnx.html
'''

x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)       # Input to the model (dummy tensor)
torch_out = torch_model(x)   #output after of the model, to verify that the model we exported computes the same values when run in ONNX Runtime.


'''
In this example we export the model with an input of batch_size 1, 
but then specify the first dimension as dynamic in the dynamic_axes parameter in torch.onnx.export(). 
The exported model will thus accept inputs of size [batch_size, 1, 224, 224] where batch_size can be variable.
'''
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
#%% check the ONNX model with ONNX’s API


import onnx

onnx_model = onnx.load("super_resolution.onnx")  #load the saved model and will output a onnx

onnx.checker.check_model(onnx_model) #verify the model’s structure and confirm that the model has a valid schema
#%% compute the output using ONNX Runtime’s Python APIs
import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")  # create an inference session for the model
                                                                     # with the chosen configuration parameters (here default config)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)  # Raises an AssertionError if two objects are not equal up to desired tolerance.
                                                                                      # match numerically with the given precision (here: rtol=1e-03 and atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
#%%
#%%Running the model on an image using ONNX Runtime


from PIL import Image
import torchvision.transforms as transforms

img = Image.open(r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\cat.jpg")

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()  #greyscale image (Y), and the blue-difference (Cb) and red-difference (Cr)

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)  # convert it to a tensor which will be the input of our model.
img_y.unsqueeze_(0)  # tensor
#%% take the tensor representing the greyscale resized cat image
# and run the super-resolution model in ONNX Runtime as explained previously.


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
#%% process the output of the model to construct back the final output image from the output tensor, and save the image.
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save(r"C:\Users\zhanq\OneDrive - Washington University in St. Louis\cat_superres_with_ort.jpg")
