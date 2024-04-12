import os
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from ultralytics import YOLO

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        # self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self.skels = []
        self.labels = []
        for i in range(1000):
            if i == 214:
                continue
            self.skels.append(os.path.join(path, str(i), 'skeleton.png'))
            self.labels.append(os.path.join(path, str(i), 'label.png'))
        self._length = len(self.skels)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        
        # self.yolo = YOLO('yolov8n-seg.pt')

    def __len__(self):
        return self._length

    def preprocess_image(self, idx):
        image = Image.open(self.skels[idx])
        weak_semantic_img = Image.open(self.labels[idx])
        # yolo_image = image
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if not weak_semantic_img.mode == "RGB":
            weak_semantic_img = weak_semantic_img.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        weak_semantic_img = np.array(weak_semantic_img).astype(np.uint8)
        weak_semantic_img = self.preprocessor(image=weak_semantic_img)["image"]
        
        # 生成弱语义标签
        # results = self.yolo(yolo_image)
        # weak_semantic_img = self.generate_weak_label(results)
        weak_img = (1.0- (weak_semantic_img / 255.)).astype(np.float32)
        
        # image = (image / 127.5 - 1.0).astype(np.float32)
        # 颠倒黑白，让骨头呈现黑色(1),背景变成 白色(0)
        image = (1.0 - (image / 255.)).astype(np.float32)
        # print(weak_img.shape)
        image = image.transpose(2, 0, 1)
        weak_img = weak_img.transpose(2, 0, 1)
        assert image.shape == weak_img.shape
        assert np.max(weak_img) == 1. and np.min(weak_img) == 0.
        return image, weak_img

    def __getitem__(self, i):
        example = self.preprocess_image(i)
        return example
    
    # def generate_weak_label(self,results):
    #     for result in results:
    #         masks = result[0].masks.data.cpu().squeeze().unsqueeze(2).numpy() # (640, 640, 1)
    #         label = plot_mask(masks) # (960,960,3)
    #     return label

def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader

# def plot_mask(mask):
#   mask = np.repeat(mask, repeats=3, axis=2)
#   assert mask.shape == (640, 640, 3)
# #   color = (127,255,0)
# #   alpha = 0.4

# #   img = np.asarray(ori_img).copy()
#   im1_shape = mask.shape
#   im0_shape = (256,256,3)

#   gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
#   pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2

#   top, left = int(pad[1]), int(pad[0])  # y, x
#   bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
#   mask = mask[top:bottom, left:right]

#   mask_img = Image.fromarray((255 * mask).astype(np.uint8))
#   # mask = np.array(mask_img.resize(im0_shape[:2])) >= 1
#   return np.array(mask_img.resize(im0_shape[:2]))
# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
