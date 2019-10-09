import os
import time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

import transforms as ext_transforms
from models.enet import ENet, ENetDepth
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

#huimin
#from data.utils import utils
from class_encoding import get_color_encoding

device = torch.device('cuda')
# Mean color, standard deviation (R, G, B)
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]
data_path = os.path.join('demo_data', 'image', '0.jpg')
depth_path = os.path.join('demo_data', 'depth', '0.png')


def predict(model, images, class_encoding):
    # measure time to upload from CPU to GPU
    start = time.time()
    images = images.to(device)
    end = time.time()
    print('upload gpu time: {} s'.format(end - start))
    # measure time for model inference
    start = time.time()
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    end = time.time()
    print('model inference time: {} s'.format(end - start))
    return predictions


def process_predict(predictions, class_encoding):
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)
    label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    return color_predictions


def setup_model():
    class_encoding= get_color_encoding('scannet20')
    num_classes = len(class_encoding)
    model = ENetDepth(num_classes).to(device)
    optimizer = optim.Adam(model.parameters())
    model = utils.load_checkpoint(model, optimizer, 'save', 'ENetDepth-scannet20')[0]
    return model, class_encoding


def scannet_loader_depth(data_path, depth_path, color_mean = [0.,0.,0.], color_std = [1.,1.,1.]):
        """Loads a sample and label image given their path as PIL images. (nyu40 classes)
        Keyword arguments:
        - data_path (``string``): The filepath to the image.
        - depth_path (``string``): The filepath to the depth png.
        - label_path (``string``): The filepath to the ground-truth image.
        - color_mean (``list``): R, G, B channel-wise mean
        - color_std (``list``): R, G, B channel-wise stddev
        - seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')
        Returns the image and the label as PIL images.
        """
        # Load image
        rgb = np.array(imageio.imread(data_path))
        # Reshape rgb from H x W x C to C x H x W
        rgb = np.moveaxis(rgb, 2, 0)
        # Define normalizing transform
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
        rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))
        # Load depth
        depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0)
        depth = torch.unsqueeze(depth, 0)
        # Concatenate rgb and depth
        data = torch.cat((rgb, depth), 0)
        return data


model, class_encoding = setup_model()
image = scannet_loader_depth(data_path, depth_path, color_mean = color_mean, color_std = color_std)
image=image.unsqueeze(dim = 0)
for i in range(10):
    predictions = predict(model, image, class_encoding)
color_predictions = process_predict(predictions, class_encoding)