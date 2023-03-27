import cv2
import numpy as np
from collections import defaultdict
import sys
import math

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
    cv2.imshow('img',des_img)

def homography():
    global img_homo
    src = np.float32(point_list)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)  # matrix 얻어옴
    img_homo = cv2.warpPerspective(src_img, matrix, (width, height))  # matrix 대로 변환
    cv2.imshow('homography', img_homo)

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """
    Group lines by their angle using k-means clustering.

    Code from here:
    https://stackoverflow.com/a/46572063/1755401
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))

    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # Get angles in [0, pi] radians
    angles = np.array([line[0][1] for line in lines])

    # Multiply the angles by two and find coordinates of that angle on the Unit Circle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype=np.float32)

    # Run k-means
    if sys.version_info[0] == 2:
        # python 2.x
        ret, labels, centers = cv2.kmeans(pts, k, criteria, attempts, flags)
    else:
        # python 3.x, syntax has changed.
        labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]

    labels = labels.reshape(-1) # Transpose to row vector

    # Segment lines based on their label of 0 or 1
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)

    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """
    Find the intersection of two lines
    specified in Hesse normal form.

    Returns closest integer pixel locations.

    See here:
    https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [[x0, y0]]

def segmented_intersections(lines):
    """
    Find the intersection between groups of lines.
    """

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def drawLines(img, lines, color=(0,0,255)):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), color, 1)

point_list = []
src_img = cv2.imread('img/tile8.jpg')
width = 1000
height = 800
color = (0,255,255)
thickness = 3
drawing = False
des_img = None

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)
cv2.waitKey(0)
cv2.destroyAllWindows()

homography()
cv2.waitKey(0)

img = img_homo
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur = cv2.medianBlur(gray, 5)
#
# # Make binary image
# adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# thresh_type = cv2.THRESH_BINARY_INV
# bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
# cv2.imshow("binary", bin_img)
# cv2.waitKey()

# Detect lines
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize=3)
rho = 1
theta = np.pi/180
thresh = 130
lines = cv2.HoughLines(edges, rho, theta, thresh)

if sys.version_info[0] == 2:
    # python 2.x
    # Re-shape from 1xNx2 to Nx1x2
    temp_lines = []
    N = lines.shape[1]
    for i in range(N):
        rho = lines[0,i,0]
        theta = lines[0,i,1]
        temp_lines.append( np.array([[rho,theta]]) )
    lines = temp_lines

# print("Found lines: %d" % (len(lines)))

# Draw all Hough lines in red
img_with_all_lines = np.copy(img)
drawLines(img_with_all_lines, lines)

cv2.imshow("Hough lines", img_with_all_lines)
cv2.waitKey()
cv2.destroyAllWindows()


# cv2.imwrite("all_lines.jpg", img_with_all_lines)

# Cluster line angles into 2 groups (vertical and horizontal)
segmented = segment_by_angle_kmeans(lines, 2)

# Find the intersections of each vertical line with each horizontal line
intersections = segmented_intersections(segmented)

img_with_segmented_lines = np.copy(img)

# Draw vertical lines in green
vertical_lines = segmented[1]
img_with_vertical_lines = np.copy(img)
drawLines(img_with_segmented_lines, vertical_lines, (0,255,0))

# Draw horizontal lines in yellow
horizontal_lines = segmented[0]
img_with_horizontal_lines = np.copy(img)
drawLines(img_with_segmented_lines, horizontal_lines, (0,255,255))

# Draw intersection points in magenta
for point in intersections:
    pt = (point[0][0], point[0][1])
    length = 5
    cv2.line(img_with_segmented_lines, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 255), 1) # vertical line
    cv2.line(img_with_segmented_lines, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 255), 1)

cv2.imshow("Segmented lines", img_with_segmented_lines)
cv2.waitKey()

### 교차점 좌표 통합
coordinates = intersections
similar_coordinates = {}
new_coordinates = []

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

## 수직선, 수평선 개수
sorted_x = sorted(new_coordinates, key=lambda x: (x[0][0], x[0][1]))  # x 좌표 오름차순 정렬
sorted_y = sorted(new_coordinates, key=lambda x: x[0][1])             # y 좌표 오름차순 정렬

for i in range(len(sorted_x)):
    if np.abs(sorted_x[i + 1][0][0] - sorted_x[i][0][0]) >= 30:
        num_horizon = i +1
        num_vertical = int(len(new_coordinates)/num_horizon)
        break

## 동일 수직선 상의 x 좌표 평균으로 통합
aver_x = []
for j in range(num_vertical):
    k = 0
    for i in range(j*num_horizon,(j+1)*num_horizon):
        k += sorted_x[i][0][0]
    aver_x.append(round(k/num_horizon))

for j in range(0,num_vertical):
    for i in range(j*num_horizon,(j+1)*num_horizon):
        sorted_x[i][0][0] = aver_x[j]

## 동일 수평선상 y좌표 평균으로 통합
aver_y = []
for j in range(0,num_horizon):
    k = 0
    for i in range(j*num_vertical,(j+1)*num_vertical):
        k += sorted_y[i][0][1]
    aver_y.append(round(k/num_vertical))

for j in range(0,num_horizon):
    for i in range(j*num_vertical,(j+1)*num_vertical):
        sorted_y[i][0][1] = aver_y[j]

# 규칙성 있는 x,y 좌표
sorted_xy = sorted(sorted_x, key=lambda x: (x[0][0], x[0][1]))

# 평균 타일 넓이 구하기
# tile_h = math.sqrt((sorted_xy[0][0][0] -sorted_xy[1][0][0]) ** 2 + (sorted_xy[0][0][1] - sorted_xy[1][0][1]) ** 2)
# for i in range(len(sorted_xy)):
#     if np.abs(sorted_xy[i + 1][0][0] - sorted_xy[i][0][0]) >= 20:
#         tile_w = math.sqrt((sorted_xy[0][0][0] -sorted_xy[i+1][0][0]) ** 2 + (sorted_xy[0][0][1] - sorted_xy[i+1][0][1]) ** 2)
#         break
sum_w = 0
sum_h = 0
for i in range(1,len(aver_x)):
    sum_w += aver_x[i] - aver_x[i-1]
aver_w = round(sum_w / (num_vertical-1))

for i in range(1,len(aver_y)):
    sum_h += aver_y[i] - aver_y[i-1]
aver_h = round(sum_h / (num_horizon-1))

aver_area = aver_h * aver_w
whole_tile = (num_vertical-1)*(num_horizon-1)


### 위 아래 조각 맞추기
if aver_y[0] < 10 and (height-aver_y[-1]) < 10 :    # 가장 윗줄 아랫줄에 쪼개진 타일 없는경우
    ud_btile = 0

elif aver_y[0] > 10 and (height-aver_y[-1]) < 10 :   # 위쪽만 있을경우
    if  aver_y[0] > aver_h / 2 :
        ud_btile = num_vertical -1
    else:
        ud_btile = round((num_vertical -1)/2)

elif aver_y[0] < 10 and (height-aver_y[-1]) > 10 : # 아래만 있을경우
    if (height-aver_y[-1]) > aver_h / 2:
        ud_btile = num_vertical - 1
    else:
        ud_btile = round((num_vertical - 1) / 2)

if aver_y[0] > 10 and (height-aver_y[-1]) > 10 :    # 둘 다 있을경우
    if  (aver_y[0] > aver_h / 2 and (height-aver_y[-1]) > aver_h / 2) and aver_y[0] + (height-aver_y[-1]) > aver_h+5:
        ud_btile = (num_vertical-1)*2

    elif ((aver_y[0] > aver_h / 2 and (height-aver_y[-1]) < aver_h / 2) and aver_y[0] + (height-aver_y[-1]) > aver_h+5) \
            or ((aver_y[0] < aver_h / 2 and (height-aver_y[-1]) > aver_h / 2)and aver_y[0] + (height-aver_y[-1]) > aver_h+5):
        ud_btile = (num_vertical-1) + round((num_vertical-1)/2)

    else :  #aver_y[0] > aver_h / 2 and (height-aver_y[-1]) < aver_h / 2:
        ud_btile = (num_vertical - 1)

# 좌,우 조각 맞추기
if aver_x[0] < 10 and (width-aver_x[-1]) < 10 :    # 왼쪽줄, 오른쪽줄 에 쪼개진 타일 없는경우
    rl_btile = 0

elif aver_x[0] > 10 and (width-aver_x[-1]) < 10 :   # 왼쪽만 있을경우
    if  aver_x[0] > aver_w / 2 :
        rl_btile = num_horizon -1
    else:
        rl_btile = round((num_horizon -1)/2)

elif aver_x[0] < 10 and (width-aver_x[-1]) > 10 : # 오른쪽만 있을경우
    if (width-aver_x[-1]) > aver_w / 2:
        rl_btile = num_horizon - 1
    else:
        rl_btile = round((num_horizon - 1) / 2)

if aver_x[0] > 10 and (width-aver_x[-1]) > 10 :    # 둘 다 있을경우
    if  (aver_x[0] > aver_w / 2 and (width-aver_x[-1]) > aver_w / 2) and aver_x[0] + (width-aver_x[-1]) > aver_w+5:
        rl_btile = (num_horizon-1)*2

    elif ((aver_x[0] > aver_w / 2 and (width-aver_x[-1]) < aver_w / 2)and aver_x[0] + (width-aver_x[-1]) > aver_w+5) or \
            ((aver_x[0] < aver_w / 2 and (width-aver_x[-1]) > aver_w / 2)and aver_x[0] + (width-aver_x[-1]) > aver_w+5):
        rl_btile = (num_horizon-1) + round((num_horizon-1)/2)

    else :  #aver_y[0] > aver_h / 2 and (height-aver_y[-1]) < aver_h / 2:
        rl_btile = (num_horizon - 1)

# 구석 조각 맞추기

if aver_x[0] < aver_w/2 and (width - aver_x[-1]) < aver_w/2 and aver_y[0] < aver_h/2 and (height - aver_y[-1]) < aver_h/2 : # 구석 조각을 온장 하나로 나눌 수 있는 경우
    an_btile = 1

elif aver_x[0] > aver_w/2 and (width - aver_x[-1]) > aver_w/2 and aver_y[0] > aver_h/2 and (height - aver_y[-1]) > aver_h/2 : # 구석 조각이 한장씩 필요한 경우
    an_btile = 4

# elif (aver_x[0] > aver_w/2 and (width - aver_x[-1]) > aver_w/2 and (aver_y[0] > aver_h/2 or (height - aver_y[-1]) > aver_h/2)) \
#         or (aver_y[0] > aver_h/2 and (height - aver_y[-1]) > aver_h/2 and (aver_x[0] > aver_w/2 or (width - aver_x[-1]) > aver_w/2)):
#     an_btile = 3
elif  ((aver_x[0] > aver_w/2 and (width - aver_x[-1]) > aver_w/2) and (aver_y[0] < aver_h/2 or (height - aver_y[-1]) < aver_h/2)) \
       or ((aver_y[0] > aver_h/2 and (height - aver_y[-1]) > aver_h/2) and (aver_x[0] < aver_w/2 or (width - aver_x[-1]) < aver_w/2)) :
    an_btile = 2

else :
    an_btile = 3

print(an_btile)
print(ud_btile)
print(rl_btile)

btile = ud_btile + rl_btile + an_btile

print("가로 줄 수 :",num_horizon)
print('세로 줄 수 :',round(num_vertical))
print("교차점 수 :",len(new_coordinates))
print(sorted_xy)
print("타일 평균높이 :",aver_h)
print('타일 평균너비 :',aver_w)
print('타일 평균면적 :',aver_area)
print('깨지지 않은 타일 수 :',whole_tile)
print('모아서 만들 수 있는 타일 수 :',btile)
print('총 사용 타일 수 :',whole_tile + btile )
print(aver_x)
print(aver_y)
cv2.imshow('intersection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

