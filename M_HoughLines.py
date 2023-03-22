import cv2
import matplotlib.pyplot as plt


def add_point(event):
    if event.inaxes != ax:  # != 같지 않다 -> 다르면 Ture
        return
    # button 1: 마우스 좌클릭
    if event.button == 1:
        x = event.xdata
        y = event.ydata
        xdata.append(x)
        ydata.append(y)
        plt.scatter(xdata, ydata)
        line.set_data(xdata, ydata)
        plt.draw()

    # button 3: 마우스 우클릭 시 기존 입력값 삭제
    if event.button == 3:
        xdata.pop()
        ydata.pop()
        line.set_data(xdata, ydata)
        plt.draw()

    # # 마우스 중간버튼 클릭 시 종료하기
    if event.button == 2:
        plt.disconnect(cid)
        plt.close()

if __name__ == "__main__":
    dir = "./Base01.jpg"
    img = cv2.imread(dir)
    fig, ax = plt.subplots(figsize=(15, 8))
    xdata = []
    ydata = []
    line, = ax.plot(xdata, ydata)
    plt.title('Interactive Plot')
    ax.set_aspect('auto', adjustable='box')
    cid = plt.connect('button_press_event', add_point)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

