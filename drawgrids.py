from PIL import Image,ImageDraw
import numpy as np
def draw_grids(img_name,no_of_rows,no_of_cols):
    img = Image.open(img_name)
    draw = ImageDraw.Draw(img)
    x_start = 0
    y_start =0
    width = img.width
    height = img.height
    x_end = width 
    y_end = height
    fill = (0,255,0) # change the color of the line in RGB format here
    x_vals = np.linspace(0,width,num= no_of_cols+1)
    for x in x_vals:
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=fill)
    line = ((x_end-1, y_start), (x_end-1, y_end))
    draw.line(line, fill=fill)
    y_vals = np.linspace(0,height,num= no_of_rows+1)
    #print(y_vals)
    for y in y_vals:
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=fill)
    line = ((x_start, y_end-1), (x_end, y_end-1))
    draw.line(line, fill=fill)
    img.show()  
    
img_path = "D:\\Education\\Others\\LPR project\\coins.jpg.png"
draw_grids(img_path,6,14) #no of rows and columns