
# coding: utf-8
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
from collections import OrderedDict

# Get the arguments
args = get_arguments()

device = torch.device(args.device)

# Mean color, standard deviation (R, G, B)
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]

def create_label_image(output, color_palette):
        """Create a label image, given a network output (each pixel contains class index) and a color palette.
        Args:
        - output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
        corresponding to the class label of that pixel.
        - color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
        """
        
        label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(color_palette):
                label_image[output==idx] = color_palette[color]
        return label_image


def predict(model, images, class_encoding):
        StartTime=time.time()
        images = images.to(device)
        # Make predictions!
        model.eval()
        with torch.no_grad():
                predictions = model(images)

        # Predictions is one-hot encoded with "num_classes" channels.
        # Convert it to a single int using the indices where the maximum (1) occurs
        _, predictions1 = torch.max(predictions.data, 1)

        label_to_rgb = transforms.Compose([
                ext_transforms.LongTensorToRGBPIL(class_encoding),
                transforms.ToTensor()
        ])
        color_predictions = utils.batch_transform(predictions1.cpu(), label_to_rgb)
        utils.imshow_batch(images.data.cpu(), color_predictions)
        EndTime=time.time()
        RunTime=EndTime-StartTime
        print("For each figure, the running time is %.4f s." %(RunTime))
    
        if args.generate_images is True:
                cur_rgb=image
                cur_output=torch.clone(predictions)
                _,cur_output=cur_output.max(0)
                cur_output = cur_output.detach().cpu().numpy()
                pred_label_image = create_label_image(cur_output, self.color_palette)
                rgb_image = image
                height = cur_output.shape[0]
                width = cur_output.shape[1]
                composite_image = np.zeros((2*height, width, 3), dtype=np.uint8)
                composite_image[0:height,:,:] = rgb_image
                composite_image[height:2*height,:,:] = pred_label_image
                imageio.imwrite(os.path.join(self.generate_image_dir, str(fileName)+'.png'), \
                                                        composite_image)
                

        
    
def scannet_loader(data_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.]):
        """Loads a sample and label image given their path as PIL images. (nyu40 classes)
        Keyword arguments:
        - data_path (``string``): The filepath to the image.
        - color_mean (``list``): R, G, B channel-wise mean
        - color_std (``list``): R, G, B channel-wise stddev
        - seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')
        Returns the image and the label as PIL images.
        """

        # Load image
        data = np.array(imageio.imread(data_path))
        # Reshape data from H x W x C to C x H x W
        data = np.moveaxis(data, 2, 0)
        # Define normalizing transform
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
        data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

        return data



def scannet_loader_depth(data_path, depth_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.]):
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
    
    
    
def get_color_encoding(seg_classes):
        if seg_classes.lower() == 'nyu40':
                """Color palette for nyu40 labels """
                return OrderedDict([
                        ('unlabeled', (0, 0, 0)),
                        ('wall', (174, 199, 232)),
                        ('floor', (152, 223, 138)),
                        ('cabinet', (31, 119, 180)),
                        ('bed', (255, 187, 120)),
                        ('chair', (188, 189, 34)),
                        ('sofa', (140, 86, 75)),
                        ('table', (255, 152, 150)),
                        ('door', (214, 39, 40)),
                        ('window', (197, 176, 213)),
                        ('bookshelf', (148, 103, 189)),
                        ('picture', (196, 156, 148)),
                        ('counter', (23, 190, 207)),
                        ('blinds', (178, 76, 76)),
                        ('desk', (247, 182, 210)),
                        ('shelves', (66, 188, 102)),
                        ('curtain', (219, 219, 141)),
                        ('dresser', (140, 57, 197)),
                        ('pillow', (202, 185, 52)),
                        ('mirror', (51, 176, 203)),
                        ('floormat', (200, 54, 131)),
                        ('clothes', (92, 193, 61)),
                        ('ceiling', (78, 71, 183)),
                        ('books', (172, 114, 82)),
                        ('refrigerator', (255, 127, 14)),
                        ('television', (91, 163, 138)),
                        ('paper', (153, 98, 156)),
                        ('towel', (140, 153, 101)),
                        ('showercurtain', (158, 218, 229)),
                        ('box', (100, 125, 154)),
                        ('whiteboard', (178, 127, 135)),
                        ('person', (120, 185, 128)),
                        ('nightstand', (146, 111, 194)),
                        ('toilet', (44, 160, 44)),
                        ('sink', (112, 128, 144)),
                        ('lamp', (96, 207, 209)),
                        ('bathtub', (227, 119, 194)),
                        ('bag', (213, 92, 176)),
                        ('otherstructure', (94, 106, 211)),
                        ('otherfurniture', (82, 84, 163)),
                        ('otherprop', (100, 85, 144)),
                ])
        elif seg_classes.lower() == 'scannet20':
                return OrderedDict([
                        ('unlabeled', (0, 0, 0)),
                        ('wall', (174, 199, 232)),
                        ('floor', (152, 223, 138)),
                        ('cabinet', (31, 119, 180)),
                        ('bed', (255, 187, 120)),
                        ('chair', (188, 189, 34)),
                        ('sofa', (140, 86, 75)),
                        ('table', (255, 152, 150)),
                        ('door', (214, 39, 40)),
                        ('window', (197, 176, 213)),
                        ('bookshelf', (148, 103, 189)),
                        ('picture', (196, 156, 148)),
                        ('counter', (23, 190, 207)),
                        ('desk', (247, 182, 210)),
                        ('curtain', (219, 219, 141)),
                        ('refrigerator', (255, 127, 14)),
                        ('showercurtain', (158, 218, 229)),
                        ('toilet', (44, 160, 44)),
                        ('sink', (112, 128, 144)),
                        ('bathtub', (227, 119, 194)),
                        ('otherfurniture', (82, 84, 163)),
                ]) 
    
    
# Run only if this module is being run directly
if __name__ == '__main__':

        # Fail fast if the dataset directory doesn't exist
#       assert os.path.isdir(
#               args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
#                       args.dataset_dir)

        # Fail fast if the saving directory doesn't exist
#       assert os.path.isdir(
#               args.save_dir), "The directory \"{0}\" doesn't exist.".format(
#                       args.save_dir)
    
    # Import the requested dataset
#       if args.dataset.lower() == 'scannet':
#               from data import ScanNet as dataset
#       else:
                # Should never happen...but just in case it does
#               raise RuntimeError("\"{0}\" is not a supported dataset.".format(
#                       args.dataset))
        
        
        class_encoding= get_color_encoding(args.seg_classes)
        num_classes = len(class_encoding)

        # Intialize a new ENet model
        
        if args.arch.lower() == 'rgb':
                model = ENet(num_classes).to(device)
        elif args.arch.lower() == 'rgbd':
                model = ENetDepth(num_classes).to(device)
        else:
                # This condition will not occur (argparse will fail if an invalid option is specified)
                raise RuntimeError('Invalid network architecture specified.')

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                                                  args.name)[0]
        # print(model)
        #inference(model, test_loader, w_class, class_encoding)
    
        image_transform = transforms.Compose(
                [transforms.Resize((args.height, args.width)),
                 transforms.ToTensor()])

        label_transform = transforms.Compose([
                transforms.Resize((args.height, args.width)),
                ext_transforms.PILToLongTensor()
        ])
    
        if args.arch.lower() == 'rgb':
            image = scannet_loader(args.data_path)
            image=image.unsqueeze(dim=0)
               # print(image.size())
        elif args.arch.lower() == 'rgbd':
            image = scannet_loader_depth(args.data_path, args.depth_path)
            image=image.unsqueeze(dim=0)
        else:
                # This condition will not occur (argparse will fail if an invalid option is specified)
            raise RuntimeError('Invalid network architecture for dataloader specified.')
        print(class_encoding)
        predict(model, image, class_encoding)



        

    

