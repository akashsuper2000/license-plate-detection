import numpy as np

def loss_function(p, a, obj, lp=1, lxy=1, lhw=0.6, lc=0.4):
    '''
        Send as Numpy Arrays
        p=predicted
        a=actual
        obj=1 if object is actually present
        vectors in order = [p, x, y, h, w, 36 classes] 
        l's are hyperparameters to control the impact of each term
    '''
    
    res = 0
    
    res+=(obj + lc(1-obj)) * (p[0]-a[0])**2
    
    res+=obj*lxy*((p[1]-a[1])**2 + (p[2]-a[2])**2)
    
    res+=obj*lhw*((p[3]-a[3])**2 + (p[4]-a[4])**2)
    
    res+=obj*lp*np.sum((p[5:]-a[5:])**2)
    
    return res



def iou(box1, box2):
    '''
    Implement the intersection over union (IoU) between box1 and box2
    
    box1 = first box, list object with coordinates (x1, y1, x2, y2)
    box2 = second box, list object with coordinates (x1, y1, x2, y2)
    '''

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    return iou
    

def find_coordinates(a, width=190, height=54):
    
    x,y,h,w = a[0],a[1],a[2],a[3]
    h*=height
    w*=width
    x*=6
    y*=14
    x+=a[-2]
    y+=a[-1]
    
    x1 = x-w/2
    y1 = y-h/2
    x2 = x+w/2
    y2 = y+h/2
    
    return [x1,y1,x2,y2]

    
def nonmax(a, th=0.2):
    '''
    Non-max suppression implementation
    a = array of p
    '''  
    c = 0
    for i in range(6):
        for j in range(14):
            w = j*13
            h = i*9
            a[c].append(w)
            a[c].append(h)
            c+=1
    
    a.sort(key = lambda i: i[0], reverse = True)
    b = []
    for p in a:
        f = 1
        for i in b:
            if(iou(find_coordinates(i[1:7]),find_coordinates(p[1:7]))>th):
                f = 0
                break
        if(f!=0):
            b.append(p)
            
    return b
    
    
    
    
    