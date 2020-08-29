#import lxml.etree as et
import xml.etree.ElementTree as ET
import math
#import numpy as np
from PIL import Image
def rotate_xml(filename, count):
    """ Parse a PASCAL VOC xml file """
    #Alpha=math.radians(Alpha)
    tree = ET.parse(filename)
    p=tree.find('path').text
    print(p)
    #objects = []
    for obj in tree.findall('object'):
        # if not check(obj.find('name').text):
        #     continue
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        
        # Read the original coordination
        x1=int(bbox.find('xmin').text)
        y1=int(bbox.find('ymin').text)
        x2=int(bbox.find('xmax').text)
        y2=int(bbox.find('ymax').text)
        img=Image.open(p)
        img.load()
        crop_img=img.crop((x1,y1,x2,y2))
        print(x1,y1,x2,y2)
        print(crop_img)
        crop_img=crop_img.resize((100,30))    
        crop_img.save("D:\\Education\\Others\\LPR project\/finalimgs\\img_"+str(count)+".jpg")
        
        '''crop_img = img[y1:y2,x1:x2]
        resized_image = cv2.resize(crop_img, (40, 40))
        print(crop_img.shape)
        #cv2.imshow("resized",resized_image)
        resized_image.show()
        cv2.waitKey(0)
        f=open("C:\\Users\\DELL\\Desktop\\Thangam\\Labelledimages\\image_0007.txt","a")
        k=resized_image.flatten()
        for j in k:
            f.write(str(j))
        f.write('\n')
        f.close()
        # Transformation
        
        print (x1, y1, x2, y2)
        tree.write(filename)'''
for i in range(1,52):
    if(i!=13 and i!=15 and i!=24 and i!=32 and i!=48 and i!=49):
        rotate_xml("D:\\Education\\Others\\LPR project\\/final\\shadow\\img_"+str(i)+".xml", i)
#rotate_xml("C:\\Users\\DELL\\Desktop\\Thangam\\Labelledimages\\image_"+str(2)+".xml", 1)   

