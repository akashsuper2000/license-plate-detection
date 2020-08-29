import numpy as np
from PIL import Image
import os

def convert_xml(filename):
    img = Image.open(filename)
    data = np.array(img, dtype = 'uint8')
    np.save(filename + '.npy', data)


train_data="D:\\Education\\Others\\LPR project\\finalimgs\\shadow\\imgn"      
train_imgs=os.listdir(train_data)

for img_path in train_imgs:
    convert_xml(train_data+"\\"+img_path)


