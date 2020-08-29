from keras import models
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw
#%%
xtrain = np.load('xarr.npy')[0:1,:,:,:]
xtrain /= 255.0
print(xtrain.shape)
ytrainorg = np.load('target.npy')
ytrain = ytrainorg[0:1,:,:,0:5]
print(ytrain.shape)
#%% model parameters
grid_x = 14
grid_y = 6
img_width = 54
img_height = 190
training_samples = ytrain.shape[0]
no_of_true = np.sum(ytrain[:,:,:,0])
#%%
def yolo_loss(y_true,y_pred):
    #shape_sor = K.shape(y_true)
    #shape = K.eval(shape_sor)
    #res+=(obj + lc(1-obj)) * (p[0]-a[0])**2
    pc_true = y_true[:,:,:,0]
    pc_pred = y_pred[:,:,:,0]
    bx_true = y_true[:,:,:,1]
    by_true = y_true[:,:,:,2]
    bx_pred = y_pred[:,:,:,1]
    by_pred = y_pred[:,:,:,2]
    w_true = y_true[:,:,:,3]
    w_pred = y_pred[:,:,:,3]
    h_true = y_true[:,:,:,4]
    h_pred = y_pred[:,:,:,4]  
    score_pred = K.sigmoid(pc_pred) 
    adjx_pred = K.sigmoid(bx_pred) 
    adjy_pred = K.sigmoid(by_pred) 
    adjw_pred = K.sigmoid(w_pred)  
    adjh_pred = K.sigmoid(h_pred)  
    #no_of_true = K.sum(pc_true)
    #class_labels = K.argmax(classes_true,-1)
    losspc = K.sum(K.square(pc_true - score_pred))/training_samples
    lossxy = K.sum(pc_true*(K.square(bx_true-adjx_pred) + K.square(by_true-adjy_pred)))/no_of_true
    
    losswh = K.sum(pc_true*(K.square(w_true-adjw_pred) + K.square(h_true-adjh_pred)))/no_of_true
    
    loss = (losspc + lossxy +losswh)
    #loss = total_loss
    
    
    return loss
#%%



model = models.load_model("modelyolo2.h5",custom_objects={'yolo_loss': yolo_loss})
#%%
ypred = model.predict(xtrain)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
np_ypred = sigmoid(np.array(ypred))
#%%
scores = np_ypred[0,:,:,0]
boxes = np_ypred[0,:,:,1:]
boxes = np.reshape(boxes,(84,4))
boxes.T[[0,1]] = boxes.T[[1,0]]
boxes.T[[2,3]] = boxes.T[[3,2]]
scores = np.reshape(scores,(84))
indices = tf.image.non_max_suppression(tf.convert_to_tensor(boxes), tf.convert_to_tensor(scores),10,iou_threshold=0.2, score_threshold=0.2)
#%%
print(indices.eval(session = K.get_session()))

img = Image.fromarray((xtrain[0,:,:,:]*255).astype('uint8'))
img.show()

#%%
y1,x1,y2,x2 = boxes[13][0], boxes[13][1], boxes[13][2], boxes[13][3]
def draw_box(img,box):
    fill = (0,255,0)
    draw = ImageDraw.Draw(img)
    x1 = box[1] * 190
    x2 = box[3] * 190
    y1 = box[0] * 54
    y2 = box[2] * 54
    draw.line(((x1,y1),(x1,y2)), fill=fill)
    draw.line(((x1,y1),(x2,y1)), fill=fill)
    draw.line(((x2,y1),(x2,y2)), fill=fill)
    draw.line(((x1,y2),(x2,y2)), fill=fill)
draw_box(img,[y1,x1,y2,x2])
#%%
img = Image.open('D:\\Education\\Others\\LPR project\\Dataset\\edited\\Mahindra-Xylo-526852d.jpg_0000_0331_0226_0176_0051.png')
img.show()
#%%
draw_box(img,[0.24074, 0.073685, 0.703704, 0.142105])
img.show()