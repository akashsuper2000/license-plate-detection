#%%
from csv import reader
import os
import numpy as np
from PIL import Image
#%%
lbl_dir = "D:\\Education\\Others\\LPR project\\labelled_change"
im_dir = "D:\\Education\\Others\\LPR project\\Dataset\\edited"
labels = os.listdir(lbl_dir)
count = len(labels) 
shape = (54,190)
#%%
x = np.zeros((count,54,190,3))
y = np.zeros((count,6,14,41))
col_val = np.linspace(0,1,num= 15)
row_val = np.linspace(0,1,num= 7)
index = 0
grid_width = 1/14
grid_height = 1/6 
for img_name in labels:
    if img_name == 'classes.txt':
        continue
    print("index is "+str(index))
    image = img_name[:-3] + 'png'
    img = Image.open(im_dir+'\\'+image).convert('RGB')
    #img.show()
    x[index,:,:,:] = np.array(img)
    file = lbl_dir+'\\'+img_name
    with open(file, newline='') as csvfile:
        rd = reader(csvfile, delimiter=' ')
        for entry in rd:
            lbl = int(entry[0])
            xc = float(entry[1])
            yc = float(entry[2])
            width = float(entry[3])
            height = float(entry[4])
            print(lbl,xc,yc,width,height)
            col = int(xc*14)
            row = int(yc*6)
            relative_x = (xc - col_val[col])/grid_width
            relative_y = (yc-row_val[row])/grid_height
            print(col,row)
            print('relative dimensions: ' + str(relative_x )+' ' +str(relative_y))
            y[index,row,col,0] =1
            y[index,row,col,1] =relative_x
            y[index,row,col,2] =relative_y
            y[index,row,col,3] =width
            y[index,row,col,4] =height
            y[index,row,col,5+lbl] =1
    index+=1
#%%
np.save('xarr.npy',x)
np.save('target.npy',y)