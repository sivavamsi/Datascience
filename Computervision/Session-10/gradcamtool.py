import torch.nn as nn
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class ModelGradCam(nn.Module):

    def __init__(self, model):
        super().__init__()

        # get the pretrained resnet network
        self.res = model

        # disect the network to access its last convolutional layer
        self.features_conv = nn.Sequential(*list(self.res.children())[:-2])

        # # get the max pool of the features stem
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the resnet
        self.classifier1 = list(self.res.children())[-2:][0]
        self.classifier2 = list(self.res.children())[-2:][1]

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.classifier1(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier2(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

def gradcam_image(model,array,threshold):
  grad_res = ModelGradCam(model)
  grad_res.eval()


  img_arr = np.transpose(array.numpy(), (1, 2, 0))
  img = ((img_arr - img_arr.min()) * (1/(img_arr.max() - img_arr.min()) * 255)).astype('uint8')
  pred = grad_res(array.unsqueeze(0))

  # get the gradient of the output with respect to the parameters of the model
  pred[:,pred.argmax().item()].backward()
  # pull the gradients out of the model
  gradients = grad_res.get_activations_gradient()
  # pool the gradients across the channels
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  # get the activations of the last convolutional layer
  activations = grad_res.get_activations(array.unsqueeze(0)).detach()
  # weight the channels by corresponding gradients

  for i in range(256):
      activations[:, i, :, :] *= pooled_gradients[i]
      # weight the channels by corresponding gradients

  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = np.maximum(heatmap.cpu(), 0)
  heatmap /= torch.max(heatmap)

  heatmap_numpy_resized = cv2.resize(heatmap.cpu().data.numpy(), (array.shape[1], array.shape[2]))
  heatmap_rescaled = np.uint8(255 * heatmap_numpy_resized)
  heatmap_final = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
  superimposed_img =  heatmap_final*threshold + img
  return superimposed_img
