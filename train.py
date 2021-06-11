# %%
import os
import numpy as np
import torch
from dataloader import *
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from dice_loss import *
import matplotlib.pyplot as plt
from unet import UNet
from unet.unet_model_cbam import UNet_cbam
from unet.model_densenet121 import CrackDense121
from unet.model_densenet201 import CrackDense201
import config
from dice_loss import dice_coeff, F1_score, dice_loss2
from tqdm import tqdm
# %%
lr = config.lr
batch_size = config.batch_size
num_epoch = config.num_epoch
Attn = config.Attn
data_dir = config.data_dir
weight_dir = config.weight_dir
loss_f = config.loss_f
optimizer = config.optimizer
if not os.path.isdir(weight_dir):
    os.mkdir(weight_dir)
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config.puzzle is not None:
    print('puzzle rotation')
    tr_transform = transforms.Compose([RandomCrop(320),PuzzleRotate(),RandomFlip(), Normalization(mean=0.5, std=0.5), ToTensor()])
else:
    print('Random Rotation')
    tr_transform = transforms.Compose([RandomCrop(320),RandomRotate(),RandomFlip(),Normalization(mean=0.5, std=0.5),ToTensor()])

val_transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
test_transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

tr_dataset = CrackDataset(root = data_dir,phase = 'train' ,transform=tr_transform)
val_dataset = CrackDataset(root = data_dir,phase = 'test' ,transform=val_transform)
#test_dataset = CrackDataset(root = data_dir,phase = 'test' ,transform=test_transform)

num_train = len(tr_dataset)
num_val = len(val_dataset)


loader_train = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
loader_val= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


num_batch_train = np.ceil(num_train / batch_size)
num_batch_val = np.ceil(num_val / batch_size)

if config.Attn == True:
    print('Attention: CBAM')
    #Net = UNet_cbam(n_channels=config.in_channels, n_classes=config.n_classes)
    Net = CrackDense201()
else:
    print('Attention: None')
    Net = UNet(n_channels=config.in_channels, n_classes=config.n_classes)
Net.to(device)

if config.loss_f == 'BCE':
    print('Loss Funtion : BCE Loss')
    #fn_loss = torch.nn.BCEWithLogitsLoss(torch.FloatTensor([5.0]).to(device))
    fn_loss = torch.nn.BCEWithLogitsLoss()
else:
    print('Loss function : Dice')
    fn_loss = dice_loss2


optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%

loss_epoch_log = []
best_score = 0
print('train data size : {}\n val data size : {}\n  model_parameter_number : {}\n'.format(num_train, num_val,count_parameters(Net)))

for epoch in tqdm(range(1 ,num_epoch+1)):
    print('Epoch {} / {}'.format(epoch, num_epoch))
    Net.train()
    loss_avg = []
    for batch, data in enumerate(tqdm(loader_train)):
        inputs = data['input'].to(device)
        labels = data['label'].to(device)

        outputs = Net(inputs)
        if config.loss_f != 'BCE':
            outputs = torch.sigmoid(outputs)
        optimizer.zero_grad()

        loss = fn_loss(outputs, labels)
        loss_avg.append(loss.item())
        loss.backward()
        optimizer.step()

        #print('epoch : {}/{} batch : {}/{} dice_coeff_loss : {}'.format(epoch, num_epoch, batch+1, int(num_batch_train), loss.item()))
    scheduler.step()            
    loss_avg = np.array(loss_avg)
    loss_epoch_log.append(np.mean(loss_avg))
    print('\nEpoch{}_loss : {} lr: {}'.format(epoch, np.mean(loss_avg),scheduler.get_lr()[0]))
    
    with torch.no_grad():
        print('Evaluation...')
        Net.eval()
        F1 = 0
        for i, val_data in enumerate(loader_val):
            inputs = val_data['input'].to(device)
            labels = val_data['label'].to(device)
            
            outputs = Net(inputs)    
            outputs = torch.sigmoid(outputs)
            f1,_,_ = F1_score(outputs, labels,device, threshold=0.5)
            #f1 = dice_coeff(outputs, labels)
            F1 += f1
        val_score = F1 / num_batch_val
        if val_score > best_score:
            best_score = val_score
            torch.save(Net.state_dict(), os.path.join(weight_dir, 'best.pth'))
            print('best wight saved!!')
        print('F1-score : {}\n'.format(val_score.item()))

print('Best validation score: ', best_score.item())

loss_epoch_log = np.array(loss_epoch_log)
np.save(os.path.join(weight_dir,'epoch_loss.npy'), loss_epoch_log)
x = np.arange(1, len(loss_epoch_log)+1)
plt.subplot(111)
plt.plot(x, loss_epoch_log)
plt.xlabel('EPOCH')
plt.ylabel('Loss')
plt.savefig(os.path.join(weight_dir,'loss_graph.png'))