import cv2
import matplotlib.pyplot as plt
import numpy as np

class InteractivePlot:
    def __init__(self, img_dir):
        self.img = cv2.imread(img_dir)
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.xdata = []
        self.ydata = []
        self.line, = self.ax.plot(self.xdata, self.ydata)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.add_point)
        self.ax.set_aspect('auto', adjustable='box')

        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def add_point(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            x = event.xdata
            y = event.ydata
            self.xdata.append(x)
            self.ydata.append(y)
            plt.scatter(self.xdata, self.ydata)
            self.line.set_data(self.xdata, self.ydata)
            plt.draw()
        if event.button == 3:
            self.xdata.pop()
            self.ydata.pop()
            self.line.set_data(self.xdata, self.ydata)
            plt.draw()
        if event.button == 2:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close()

        self.update_coord()

    def show(self):
        plt.show()

    def update_coord(self):
        self.coord = InteractivePlot.coord(self.xdata, self.ydata)

    class coord:
        def __init__(self, x, y):
            self.x = np.array(x)
            self.y = np.array(y)
            self.xy_data = np.stack((self.x, self.y), axis=1)


if __name__ == "__main__":
    dir = "./Base01.jpg"
    plot = InteractivePlot(dir)
    plot.show()
    print(plot.xdata)
    print(plot.ydata)
    print(plot.coord.x)
