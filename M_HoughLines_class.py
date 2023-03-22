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

    def show(self):
        plt.show()
    class coord:
        def __init__(self):
            self.x = np.array(InteractivePlot.xdata)
            self.y = np.array(InteractivePlot.ydata)
            self.xy = np.stack((self.x, self.y), axis=1)

        # def coord_sort():
        #     x = np.array(x)
        #     k = x[:, 0]
        #     s = k.argsort()
        #     centers_sorted = x[s]
        #     for i in range(len(centers_sorted) // 2):
        #         b = centers_sorted[2 * i:2 * (i + 1), :]
        #         k = b[:, 1]
        #         s = k.argsort()
        #         centers_sorted[2 * i:2 * (i + 1), :] = b[s]
        #     return centers_sorted


if __name__ == "__main__":
    dir = "./Base01.jpg"
    plot = InteractivePlot(dir)
    plot.show()
    print(plot.xdata)
    print(plot.ydata)
    print(plot.coord)