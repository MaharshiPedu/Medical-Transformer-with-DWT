import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pywt
from PIL import Image
# parent_dir = '/mnt/d/Documents/Research/disWT/T1fusion_out/'
# for k in range(3):
#     img_folder = '{}'.format(k) + '_out/'
#     for i in range(2):
        
#         sub_directory = '{}_{}'.format(k, i) + '_out/'
#         for j in range(2):
#             _file = '{}_{}_{}'.format(k, i, j)
#             path = os.path.join(parent_dir, img_folder, sub_directory, _file)
#             os.makedirs(path)

img_temp = mpimage.imread('T1fusion/1/IMG-0004-00002IMG-0004-00001.png')
coeff2 = pywt.dwt2(img_temp, 'db3', mode='periodization')
LL, (LH, HL, HH) = coeff2

for i, a in enumerate([LL, LH, HL, HH]):
    plt.figure(figsize=(1,1), dpi=256)
    plt.imshow(a, interpolation='nearest', cmap=plt.cm.gray)
    # plt.gca().set_axis_off()
    plt.axis('off')
    # plt.margins(0,0)
    plt.tight_layout(pad=0)
    plt.savefig('plot.png')
    
    

idwt_fig = pywt.idwt2(coeff2, 'db3')

plt.figure(figsize=(1,1), dpi=256)
plt.imshow(idwt_fig, interpolation='nearest', cmap=plt.cm.gray)
# plt.gca().set_axis_off()
plt.axis('off')
# plt.margins(0,0)
plt.tight_layout(pad=0)
plt.savefig('idwt_out.png')


