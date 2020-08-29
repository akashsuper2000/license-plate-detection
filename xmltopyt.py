from string import ascii_lowercase
import xml.etree.ElementTree as ET
import os
from PIL import Image
cntf=0
cntnf=0
a = list(range(10))
for i in a :
   a[i] = chr(ord('0')+(a[i]))
a1 = a+ list(ascii_lowercase)
count = dict((el,0) for el in a1)

def convert_xml(filename):
    """ Parse a PASCAL VOC xml file """
    
    tree = ET.parse(filename)
    p=tree.find('path').text
    #print(p)
    #objects = []
    for obj in tree.findall('object'):
        # if not check(obj.find('name').text):
        #     continue
        global cntf,cntnf
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #print(obj_struct['name'])
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
        img=img.convert('RGB')
        crop_img=img.crop((x1,y1,x2,y2))
        #print(crop_img.size)
        '''crop_img=crop_img.resize((10,20))
        print(crop_img.size)'''
        '''
        if obj_struct['name']=='fire':
            #f1=open("H:\\Svm firedetection data\\nfimg.txt",'a')
            crop_img.save("D:\\smart space lab\\testposimg"+str(cntf)+".jpg")
            #f1.write(str(cntnf) + " "+ p)
            #f1.write("\n")
            cntf+=1
        else:
            #f1=open("H:\\Svm firedetection data\\fimg.txt",'a')
            #crop_img.save("H:\\final cropped images\\neg\\img"+str(cntnf)+".jpg")
            #f1.write(str(cntf) + " "+ p)
            #f1.write("\n")
           # cntnf+=1
           crop_img.save("D:\\Education\\Others\\LPR project\\finalimgs\\img"+str(cntnf)+".jpg")
           cntnf+=1
           #crop_img.show()'''
        '''
        print (x1, y1, x2, y2)
        tree.write(filename)'''
        #print(obj_struct['name'])
        crop_img.save("D:\\Education\\Others\\LPR project\\finalimgs\\img"+obj_struct['name']+"\\img"+str(count[obj_struct['name']])+".jpg")
        count[obj_struct['name']] +=1
train_data="D:\\Education\\Others\\LPR project\\final\\shadow"      
train_imgs=os.listdir(train_data)
for img_path in train_imgs:
    convert_xml(train_data+"\\"+img_path)
#convert_xml("F:\\lab fire annotations\\7550.xml")   

#convert_xml("D:\\smart space lab\\psodataset\\20130611_100618_Black-Forest-Fire-engulfs-house.xml")