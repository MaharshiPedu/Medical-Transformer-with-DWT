# Code for MedT

import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
import disWT.discrete_wt as dwt


parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)

# DataLoader combines a dataset and a sampler and returns an iterable over the given dataset
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)

'''Checking if more than 1 GPUs are present'''
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

criterion = LogNLLLoss()
#Optimization is the process of adjusting model parameters to reduce model error in each training step.
# parameters are all the attributes associated with the model object. 
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)  


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed)


for epoch in range(args.epochs):

    epoch_running_loss = 0
    
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader): # X_batch represents dataset and y_batch represents label     
        '''
        1. Here, we need to find the dwt of every image in the dataset i.e LL LH HL HH bands
        2. Pass LL to the model. <output_LL = model(LL)>
        3. Pass LH to the model. <output_LH = model(LH)>   
        4. Pass HL to the model. <output_HL = model(HL)>
        5. Pass HH to the model. <output_HH = model(HH)>

        6. Take idwt of all these outputs. <idwt_out_fig = pywt.idwt2(coeff2, 'db3')>
        7. Get the loss function of this.
        '''
        output_bands_train = dwt.DWT(X_batch)

        # converting the received list of numpy arrays to a numpy array.
        output_bands_train = np.array(output_bands_train)

        # Storing the individual numpy arrays of the bands.
        LL_train = output_bands_train[0]
        LH_train = output_bands_train[1]
        HL_train = output_bands_train[2]
        HH_train = output_bands_train[3]

        #Converting the individual numpy arrays of bands to tensor before wrapping them as Variable
        LL_train = torch.tensor(LL_train)
        LH_train = torch.tensor(LH_train)
        HL_train = torch.tensor(HL_train)
        HH_train = torch.tensor(HH_train)

        # wrapping the individual numpy arrays of bands into Variable
        LL_train = Variable(LL_train.to(device ='cuda'))
        LH_train = Variable(LH_train.to(device ='cuda'))
        HL_train = Variable(HL_train.to(device ='cuda'))
        HH_train = Variable(HH_train.to(device ='cuda'))

        y_batch = Variable(y_batch.to(device='cuda'))
        
        # ===================forward=====================
        
        output_LL = model(LL_train)
        output_LH = model(LH_train)
        output_HL = model(HL_train)
        output_HH = model(HH_train)
        #output = model(X_batch)  # Output from the transformer

        tmp2 = y_batch.detach().cpu().numpy() # detach().cpu().numpy() this combination of method calls detaches the gpu, assigns cpu and converts the tensor to a numpy array 
        tmpLL = output_LL.detach().cpu().numpy()
        tmpLH = output_LH.detach().cpu().numpy()
        tmpHL = output_HL.detach().cpu().numpy()
        tmpHH = output_HH.detach().cpu().numpy()

        # Applying masks' color, black or white
        tmpLL[tmpLL>=0.5] = 1
        tmpLL[tmpLL<0.5] = 0

        tmpLH[tmpLH>=0.5] = 1
        tmpLH[tmpLH<0.5] = 0

        tmpHL[tmpHL>=0.5] = 1
        tmpHL[tmpHL<0.5] = 0

        tmpHH[tmpHH>=0.5] = 1
        tmpHH[tmpHH<0.5] = 0

        tmp2[tmp2>0] = 1
        tmp2[tmp2<=0] = 0

        tmp2 = tmp2.astype(int)
        tmpLL = tmpLL.astype(int)
        tmpLH = tmpLH.astype(int)
        tmpHL = tmpHL.astype(int)
        tmpHH = tmpHH.astype(int)

        yHaT_LL = tmpLL
        yHaT_LH = tmpLH
        yHaT_HL = tmpHL
        yHaT_HH = tmpHH
        yval = tmp2

        #Taking IDWT
        output = dwt.IDWT(output_LL, output_LH, output_HL, output_HH)

        #Converting the output from numpy array to tensor
        output = torch.tensor(output)

        #wrapping the output from in a variable
        output = Variable(output.to(device ='cuda'))
        

        loss = criterion(output, y_batch)
        
        # ===================backward====================
        optimizer.zero_grad() # zero_grad() zeroes all the gradients accumulated so far.
        loss.backward()
        optimizer.step() #After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
        epoch_running_loss += loss.item()
        
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))

    
    if epoch == 10:
        for param in model.parameters():
            param.requires_grad =True
    if (epoch % args.save_freq) ==0:  # After every 10 epochs we use the validation dataset for improvement in learning. This ensures that the model is actually 'learning' and not "remembering"

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            # print(batch_idx)
            if isinstance(rest[0][0], str): # checking if rest[0][0] is a string.
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(batch_idx + 1).zfill(3) #zfill(3) adds preceding zeroes to the name of the image until the length of the name becomes 3.

            output_bands_val = dwt.DWT(X_batch)

            # converting the received list of numpy arrays to a numpy array.
            output_bands_val = np.array(output_bands_val)

            # Storing the individual numpy arrays of the bands.
            LL_val = output_bands_val[0]
            LH_val = output_bands_val[1]
            HL_val = output_bands_val[2]
            HH_val = output_bands_val[3]

            #Converting the individual numpy arrays of bands to tensor before wrapping them as Variable
            LL_val = torch.tensor(LL_val)
            LH_val = torch.tensor(LH_val)
            HL_val = torch.tensor(HL_val)
            HH_val = torch.tensor(HH_val)

            # wrapping the individual numpy arrays of bands into Variable
            LL_val = Variable(LL_val.to(device ='cuda'))
            LH_val = Variable(LH_val.to(device ='cuda'))
            HL_val = Variable(HL_val.to(device ='cuda'))
            HH_val = Variable(HH_val.to(device ='cuda'))

            y_batch = Variable(y_batch.to(device='cuda'))

            output_LL_val = model(LL_val)
            output_LH_val = model(LH_val)
            output_HL_val = model(HL_val)
            output_HH_val = model(HH_val)
            #y_out = model(X_batch)

            # start = timeit.default_timer()
           
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            tmp2 = y_batch.detach().cpu().numpy()
            tmpLL = output_LL_val.detach().cpu().numpy()
            tmpLH = output_LH_val.detach().cpu().numpy()
            tmpHL = output_HL_val.detach().cpu().numpy()
            tmpHH = output_HH_val.detach().cpu().numpy()

            # Applying masks' color, black or white
            tmpLL[tmpLL>=0.5] = 1
            tmpLL[tmpLL<0.5] = 0

            tmpLH[tmpLH>=0.5] = 1
            tmpLH[tmpLH<0.5] = 0

            tmpHL[tmpHL>=0.5] = 1
            tmpHL[tmpHL<0.5] = 0

            tmpHH[tmpHH>=0.5] = 1
            tmpHH[tmpHH<0.5] = 0

            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0

            tmp2 = tmp2.astype(int)
            tmpLL = tmpLL.astype(int)
            tmpLH = tmpLH.astype(int)
            tmpHL = tmpHL.astype(int)
            tmpHH = tmpHH.astype(int)

            yHaT_LL = tmpLL
            yHaT_LH = tmpLH
            yHaT_HL = tmpHL
            yHaT_HH = tmpHH
            yval = tmp2


            # print(np.unique(tmp2))

            epsilon = 1e-20
            
            del X_batch, y_batch,tmpLL, tmpLH, tmpHL, tmpHH, tmp2, output_LL_val, output_LH_val, output_HL_val, output_HH_val
 
            
            yHaT_LL[yHaT_LL==1] =255
            yHaT_LH[yHaT_LH==1] =255
            yHaT_HL[yHaT_HL==1] =255
            yHaT_HH[yHaT_HH==1] =255
            yval[yval==1] =255
            fulldir = direc+"/{}/".format(epoch)
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):
                
                os.makedirs(fulldir)
            
            cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])
            # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
        fulldir = direc+"/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir+args.modelname+".pth")
        torch.save(model.state_dict(), direc+"final_model.pth")
            


