import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import cv2

def txt_coord(dir, nc):
    line_pd = pd.read_csv(dir , sep= " ", header=None) # 첫째행이 header가 아닌경우 header = None
    mask = line_pd[0] == nc # class number 추출
    filtered_data =line_pd[mask]
    filtered_data = filtered_data.to_numpy()
    shp = filtered_data.shape
    filtered_data = filtered_data[:, 1:].reshape(shp[0],-1,2)
    filtered_data = filtered_data
    return filtered_data

def draw_tile(coords):
    global width
    global height
    area = []
    for k in range(len(coords)):
        x = coords[k][:,0]
        x_mask = np.isnan(x) # nan 있는 곳 추출
        x = x[~x_mask] *width # nan 반대 ~nan
        y = coords[k][:,1]
        y_mask = np.isnan(y)
        y= y[~y_mask] *height
        plt.plot(x, y)
        s = map(Point, zip(x,y))
        poly = Polygon(s)
        center = poly.centroid
        plt.text(center.x, center.y, round(poly.area, 2), fontsize="xx-small")
        area.append(round(poly.area, 2))
    return area

if __name__ == "__main__":
    img_dir = "./49.jpg"
    txt_dir = "./49.txt"
    img = cv2.imread(img_dir)
    width = np.shape(img)[0]
    height = np.shape(img)[1]

    coords = txt_coord(txt_dir, 0)
    area = draw_tile(coords)
    plt.imshow(img)
    plt.show()
    print(area)