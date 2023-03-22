'''
tile이미지 불러오기 0
좌표 찍기0
호모그래피0
허프라인 0
intersection 0
intersection 거리
넓이
'''기
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import math


point_list = []
src_img = cv2.imread('img/tile8.jpg')
src_img
width = 1000
height = 800

color = (255,0,255)
thickness = 3
drawing = False
des_img = None

def mouse_handler(event, x,y,flags,param) :
    global drawing, des_img
    des_img = src_img.copy()
    if event == cv2.EVENT_LBUTTONDOWN :
        drawing = True
        point_list.append((x,y))

    if drawing :
        prev_point = None   # 직선의 시작점
        for point in point_list :
            cv2.circle(des_img,point,8, color, cv2.FILLED)
            if prev_point :
                cv2.line(des_img,prev_point,point,color, thickness,cv2.LINE_AA)
            prev_point = point

        next_point = (x,y)
        if len(point_list) == 4 :
            next_point = point_list[0]   # 첫 번째 시작점
            cv2.line(des_img,prev_point,next_point,color, thickness,cv2.LINE_AA)
            show_result()

    cv2.imshow('img',des_img)


def show_result() :
    src = np.float32(point_list)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)  # matrix 얻어옴
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    homography = cv2.warpPerspective(src_img, matrix, (width, height))  # matrix 대로 변환
    gray_homography = cv2.warpPerspective(gray, matrix, (width, height))

    rows = 2
    cols = 2
    fig = plt.figure(figsize=(6, 4))

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(cv2.cvtColor(des_img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(cv2.cvtColor(homography, cv2.COLOR_BGR2RGB))
    ax2.set_title('Homography')
    ax2.axis("off")

    gray = cv2.cvtColor(homography, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)


    img = homography

    def segment_by_angle_kmeans(lines, k=2, **kwargs):

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

        flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)

        # Get angles in [0, pi] radians
        angles = np.array([line[0][1] for line in lines])

        # Multiply the angles by two and find coordinates of that angle on the Unit Circle
        pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32)

        # Run k-means
        if sys.version_info[0] == 2:
            # python 2.x
            ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
        else:
            # python 3.x, syntax has changed.
            labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

        labels = labels.reshape(-1)  # Transpose to row vector

        # Segment lines based on their label of 0 or 1
        segmented = defaultdict(list)
        for i, line in zip(range(len(lines)), lines):
            segmented[labels[i]].append(line)

        segmented = list(segmented.values())

        return segmented

    def intersection(line1, line2):

        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))

        return [[x0, y0]]

    def segmented_intersections(lines):

        intersections = []
        for i, group in enumerate(lines[:-1]):
            for next_group in lines[i + 1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(intersection(line1, line2))
        return intersections

    def drawLines(img, lines, color=(0, 0, 255)):

        for line in lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

    segmented = segment_by_angle_kmeans(lines, 2)
    intersections = segmented_intersections(segmented)
    img_with_segmented_lines = np.copy(img)
    vertical_lines = segmented[1]

    drawLines(img_with_segmented_lines, vertical_lines, (0, 255, 0))
    horizontal_lines = segmented[0]
    drawLines(img_with_segmented_lines, horizontal_lines, (0, 255, 255))

    coordinates = intersections

    similar_coordinates = {}
    global new_coordinates
    new_coordinates = []

    ### 교차점 좌표 통합
    for coord in coordinates:
        similar = False
        for k in similar_coordinates:
            if math.sqrt((coord[0][0] - k[0]) ** 2 + (coord[0][1] - k[1]) ** 2) <= 50:
                similar_coordinates[k].append(coord)
                similar = True
                break
        if not similar:
            similar_coordinates[tuple(coord[0])] = [coord]

    for k, v in similar_coordinates.items():
        x = sum([coord[0][0] for coord in v]) / len(v)
        y = sum([coord[0][1] for coord in v]) / len(v)
        new_coordinates.append([[int(x), int(y)]])

    for point in new_coordinates:
        pt = (point[0][0], point[0][1])
        length = 5
        cv2.line(img, (pt[0], pt[1] - length), (pt[0], pt[1] + length), (255, 0, 255),5)  # vertical line
        cv2.line(img, (pt[0] - length, pt[1]), (pt[0] + length, pt[1]), (255, 0, 255), 5)

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(cv2.cvtColor(img_with_segmented_lines, cv2.COLOR_BGR2RGB))
    ax3.set_title('Hough Line Transform')
    ax3.axis("off")




    ax4 = fig.add_subplot(rows, cols, 4)
    ax4.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax4.set_title('inter_section')
    ax4.axis("off")

    sorted_x = sorted(new_coordinates, key=lambda x: (x[0][0], x[0][1]))
    print("교차점 수 :",len(new_coordinates))
    print(sorted_x)

    tile_h = math.sqrt((sorted_x[0][0][0] -sorted_x[1][0][0]) ** 2 + (sorted_x[0][0][1] - sorted_x[1][0][1]) ** 2)
    print("타일 높이 :",tile_h)
    for i in range(len(sorted_x)-1):
        if np.abs(sorted_x[i + 1][0][0] - sorted_x[i][0][0]) >= 20:
            # print(sorted_x[i + 1])

            tile_w = math.sqrt((sorted_x[0][0][0] -sorted_x[i+1][0][0]) ** 2 + (sorted_x[0][0][1] - sorted_x[i+1][0][1]) ** 2)
            break
    # sorted_y = sorted(new_coordinates, key=lambda x: (x[0][1], x[0][0]))
    # print(sorted_y)
    # tile_w = math.sqrt((sorted_y[0][0][0] -sorted_y[1][0][0]) ** 2 + (sorted_y[0][0][1] - sorted_y[1][0][1]) ** 2)
    # print('타일 넓이 :',tile_w)
    print('타일 너비 : ',tile_w)
    area = tile_h * tile_w
    print('타일 면적 :',area)

    plt.show()
    plt.close()

cv2.namedWindow('img')
cv2.setMouseCallback('img',mouse_handler)
cv2.imshow('img',src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print()
