import os

def convert(a, w=190, h=54):
    x1,x2 = a[1]-a[3]/2, a[1]+a[3]/2
    y1,y2 = a[2]-a[4]/2, a[2]+a[4]/2
    
    return [int(a[0]),round(x1,6),round(y1,6),round(x2,6),round(y2,6)]

path = 'D:\\Education\\Others\\LPR project\\Dataset_labelled\\'
write_path = 'D:\\Education\\Others\\LPR project\\labelled_change\\'
files = os.listdir(path)
for i in files:
    if(i!='classes.txt'):
        f = open(path+i,'r')
        f1 = f.readlines()
        f2 = open(write_path+i, 'w+')
        for j in f1:
            a = [float(k) for k in j.split()]
            ans = convert(a)
            f2.write(' '.join([str(m) for m in ans]))
            f2.write('\n')
        f2.close()
        f.close()
