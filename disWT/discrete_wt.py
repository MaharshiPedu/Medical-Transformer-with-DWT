from numpy.core.fromnumeric import size
import pywt
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import glob
import os
from torch.utils.data import DataLoader
import numpy as np
import torch
def DWT(x_batch):
    x_batch_np = x_batch.detach().cpu().numpy()
    LL_list, LH_list, HL_list, HH_list = [], [], [], []
    for _ in range(len(x_batch_np)): # len(x_batch_np) finds out the number of images in the dataset
    
        imgs = [mpimage.imread(file) for file in x_batch] #Reading all the images in a particular subdirectory, say, 1 in T1fusion
        tensor_imgs = torch.tensor(imgs)
        for img in tensor_imgs:
            
            coeffs2 = pywt.dwt2(img, 'db3')
            LL, (LH, HL, HH) = coeffs2
            LL_list.append(LL)
            LH_list.append(LH)
            HL_list.append(HL)
            HH_list.append(HH)

    LL_list = np.array(LL_list)
    LH_list = np.array(LH_list)
    HL_list = np.array(HL_list)
    HH_list = np.array(HH_list)
    
    return [LL_list, LH_list, HL_list, HH_list]

def IDWT(output_LL, output_LH, output_HL, output_HH):
    output_LL_np = output_LL.detach().cpu().numpy()
    output_LH_np = output_LH.detach().cpu().numpy()
    output_HL_np = output_HL.detach().cpu().numpy()
    output_HH_np = output_HH.detach().cpu().numpy()

    coeff2 = output_LL_np, (output_LH_np, output_HL_np, output_HH_np)
    idwt_fig = pywt.idwt2(coeff2, 'db3')
    return idwt_fig
        
# new_path = os.path.join(parent_dir, img_folder, sub_directory)
# os.makedirs(new_path)
# for j, a in enumerate([LL, LH, HL, HH]):
#     plt.figure(figsize=(1, 1), dpi=256)
    
#     plt.imshow(a, interpolation='nearest', cmap=plt.cm.gray)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.savefig(new_path +'{}_{}_{}'.format(k+1, i+1, j+1)  +'.png')
    
#     plt.show()