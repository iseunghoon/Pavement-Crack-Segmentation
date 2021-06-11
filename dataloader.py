import os
import numpy as np
import torch
import cv2
import torch.utils.data.dataset
from torchvision import transforms, datasets
import glob
from PIL import Image


class CrackDataset(object):
    def __init__(self, root ,phase,transform=None):
        self.root = root
        self.images = sorted(glob.glob(os.path.join(self.root,phase,'images')+'/*'))
        self.labels = sorted(glob.glob(os.path.join(self.root, phase, 'masks_bin')+'/*'))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        self.img_path = self.images[idx]
        self.label_path = self.labels[idx]
        img = cv2.imread(self.img_path)
        label = cv2.imread(self.label_path, 0)
        #label = label[:,:,np.newaxis]
        data = {'input': img, 'label':label}
        
        if self.transform is not None:
            data = self.transform(data)
                    
        return data
    
    def get_image_path(self, index):
        return self.images[index]

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = label[:,:,np.newaxis]
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'] / 255, data['input'] / 255

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        label, input = data['label'], data['input']

        h, w = input.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h+1)
        left = np.random.randint(0, w - new_w+1)

        input = input[top: top + new_h,
                      left: left + new_w]

        label = label[top: top + new_h,
                      left: left + new_w]
        data = {'label': label, 'input': input}

        return data

class ColorJitter(object):
    def __init__(self, thres):
        self.thres = thres
    def __call__(self, data):
        label, input = data['label'], data['input']
        
        if np.random.rand() > 0.6:
            input = Image.fromarray(input)
            aug_f = transforms.ColorJitter(contrast=self.thres, brightness=self.thres)
            input = aug_f(input)
        
        data = {'label': label, 'input': np.array(input)}
        
        return data

class RandomRotate(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.3:
            degree = np.random.randint(0,360)
            rows,cols = label.shape[:2]
            M= cv2.getRotationMatrix2D((cols/2, rows/2),degree,1.0)
            dst_img = cv2.warpAffine(input, M,(cols, rows))
            dst_label = cv2.warpAffine(label, M,(cols, rows))
            _, thres = cv2.threshold(dst_label,128,255,cv2.THRESH_BINARY)
            data = {'label':thres, 'input':dst_img}
        return data

            
class PuzzleRotate(object): #puzzle 16x16 
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.3:
            degree = np.random
            puzzle_size = input.shape[1] // 4
            h,w,c = input.shape
            puzzle_img = np.zeros((h,w,c),dtype=np.uint8)
            puzzle_label = np.zeros((h,w))
            for i in range(4):
                for j in range(4):
                    degree = np.random.randint(0,360)
                    #print(degree)
                    r_img = input[i*puzzle_size:(i+1)*puzzle_size,j*puzzle_size:(j+1)*puzzle_size]
                    r_label = label[i*puzzle_size:(i+1)*puzzle_size,j*puzzle_size:(j+1)*puzzle_size]
                    rows,cols = r_label.shape[:2]
                    M= cv2.getRotationMatrix2D((cols/2, rows/2),degree,1.0)
                    dst_img = cv2.warpAffine(r_img, M,(cols, rows))
                    #crop_mask = dst_img == 0
                    #dst_img[crop_mask] = np.average(dst_img[~crop_mask])
                    dst_label = cv2.warpAffine(r_label, M,(cols, rows))
                    _, thres = cv2.threshold(dst_label,128,255,cv2.THRESH_BINARY)
                    puzzle_img[i*puzzle_size:(i+1)*puzzle_size,j*puzzle_size:(j+1)*puzzle_size,:] = dst_img[:,:,:3]
                    puzzle_label[i*puzzle_size:(i+1)*puzzle_size,j*puzzle_size:(j+1)*puzzle_size] = thres
            data = {'label':puzzle_label, 'input':puzzle_img}
        return data
