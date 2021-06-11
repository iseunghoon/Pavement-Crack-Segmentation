##learning parameters

batch_size = 4
in_channels = 3
n_classes = 1
lr = 1e-3
num_epoch = 100
data_dir = "/home/mvpserver20/data/CrackForest-dataset"
loss_f = 'DiCE'
optimizer = 'Adam'
weight_dir = './weight_CFD_basedice'
puzzle = None
Attn = True
###Test

weight_path = "/home/mvpserver20/data/shlee/Pytorch-UNet/weight_CFD_densenet201/best.pth"
#weight_path = "/home/mvpserver20/data/shlee/Pytorch-UNet/weight_CFD_basedice/best.pth" # "/home/mvpserver20/data/shlee/Pytorch-UNet/weight_CFD_densenet201/best.pth"
eval = "F1"