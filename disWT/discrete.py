import numpy as np
import pywt
import pywt.data
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import glob
from PIL import Image


# # imgs = [mpimage.imread(file) for file in glob.glob('T1fusion/1/*.png')]
# img_temp = mpimage.imread('T1fusion/1/IMG-0004-00002IMG-0004-00001.png')
# # for img in imgs:
# coeff2 = pywt.dwt2(img_temp, 'db3', mode='periodization') # dwt
# LL, (LH, HL, HH) = coeff2 # Extracting coefficients Approximate, (horizontal, vertical, diagonal)

# '''
# This part can be used for inverse dwt.
# '''

# # imgr = pywt.idwt2(coeff2, 'db3', mode='periodization') # idwt
# # imgr = np.uint8(imgr)

# #Plotting wavelet coefficients

# fig = plt.figure(figsize=(30,30))

# rows = 1
# columns = 4
# titles = ['LL', 'LH', 'HL', 'HH']

# # out_path = 'T1fusionOutput/1_out/'

# count = 1;

# for i, a in enumerate([LL, LH, HL, HH]):
#     j=1
#     # ax = fig.add_subplot(2, 2, i+1)

#     plt.imshow(a, cmap=plt.cm.gray)
#     #plt.title(titles[i], fontsize=40)
#     plt.axis('off')
    
#     # output_img_name = 'plot_'+'{}_{}'.format(count, j) +'.png'
#     # plt.savefig(os.path.join(out_path, output_img_name), dpi=500)
#     plt.savefig('plot.png', dpi=500)
#     # count += 1
#     j += 1
#     plt.show()

