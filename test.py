# %%
import os
from dataloader import *
import numpy as np
#from dice_loss import dice_coeff, F1_score
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import transforms
import time
from unet import UNet
from unet.unet_model_cbam import UNet_cbam
from unet.model_densenet201 import CrackDense201
from tqdm import tqdm
from dice_loss import dice_coeff, F1_score, dice_loss2
import config

data_dir = config.data_dir
weight_path = config.weight_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
test_data = CrackDataset(root=data_dir, phase='test', transform=transform)
loader_test = DataLoader(test_data, batch_size=1, shuffle=False)

if config.Attn == True:
    #Net = UNet_cbam(n_channels=config.in_channels, n_classes=config.n_classes)
    Net = CrackDense201()
else:
    Net = UNet(n_channels=config.in_channels, n_classes=config.n_classes)
Net.to(device)
Net.load_state_dict(torch.load(weight_path))
Net.eval()

    
if config.eval == "OIS":
    print("OIS")
    average_score = []
    threshold_list = np.arange(1,100) / 100
    for i, data in enumerate(tqdm(loader_test)):
        images = data['input'].to(device)
        labels = data['label'].to(device)
        outputs = Net(images)
        best_score = 0
        for th in threshold_list:
            F1, Pr, Re = F1_score(outputs, labels, device=device, threshold = th)
            if F1.item() > best_score:
                best_score = F1.item()
        #print(best_score)
        average_score.append(best_score)
    average_score = np.array(average_score)
    print("Evaluation result \nF1_score : {}".format(np.mean(average_score)))
elif config.eval == "ODS":
    print("ODS")
    threshold_list = np.arange(1,100) / 100
    Best_score = []
    for th in tqdm(threshold_list):
        #print('Threshold : {}'.format(th))
        average_Pr = []
        average_Re = []
        for i, data in enumerate(loader_test):
            #print("{} / {}".format(i+1, len(test_data)))
            images = data['input'].to(device)
            labels = data['label'].to(device)
            outputs = Net(images)
            F1, Pr, Re = F1_score(outputs, labels, device=device, threshold = th)
            average_Pr.append(Pr.item())
            average_Re.append(Re.item())
        average_Pr = np.array(average_Pr)
        avg_Pr = np.mean(average_Pr)
        average_Re = np.array(average_Re)
        avg_Re = np.mean(average_Re)
        F1 = 2*avg_Pr*avg_Re / (avg_Pr+avg_Re)
        #print('F1_score : {}'.format(F1))
        Best_score.append(F1)
    Best_score = np.max(np.array(Best_score))
    print("Evaluation result \nF1_score : {}".format(Best_score))
elif config.eval == "F1":
    print("F1-score")
    avg_F1 = []
    avg_pr = []
    avg_re = []
    for i, data in enumerate(tqdm(loader_test)):
        #print("{} / {}".format(i+1, len(test_data)))
        images = data['input'].to(device)
        labels = data['label'].to(device)
        outputs = Net(images)
        outputs = torch.sigmoid(outputs)

        F1, Pr, Re = F1_score(outputs, labels, device=device, threshold = 0.5)
        print(F1)
        avg_F1.append(F1.item())
        avg_pr.append(Pr.item())
        avg_re.append(Re.item())

    avg_F1 = np.array(avg_F1)
    avg_pr = np.array(avg_pr)
    avg_re = np.array(avg_re)
    #print(np.mean(avg_F1))
    print('F1_score : {}\nPrecision : {}\nRecall : {}'.format(np.mean(avg_F1), np.mean(avg_pr),np.mean(avg_re)))