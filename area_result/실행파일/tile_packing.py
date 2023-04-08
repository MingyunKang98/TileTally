import cv2
import numpy as np
from collections import defaultdict
import sys
import math
from rectpack import newPacker, PackingMode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def sort_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = np.sum(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def homography():
    global img_homo
    src = np.float32(point_list)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)  # matrix 얻어옴
    img_homo = cv2.warpPerspective(src_img, matrix, (width, height))  # matrix 대로 변환
    cv2.imshow('homography', img_homo)
    cv2.imwrite("img_homo.jpg", img_homo)


def mouse_handler(event, x, y, flags, param):
    global point_list, des_img, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append((x, y))
        drawing = True

    if event == cv2.EVENT_MOUSEMOVE and drawing:
        des_img = src_img.copy()
        cv2.circle(des_img, point_list[-1], 10, color, thickness)

        if len(point_list) > 1:
            cv2.line(des_img, point_list[-2], point_list[-1], color, thickness)

        cv2.imshow('img', des_img)

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        des_img = src_img.copy()
        cv2.circle(des_img, point_list[-1], 10, color, thickness)

        if len(point_list) > 1:
            prev_point = point_list[0]
            for point in point_list[1:]:
                cv2.line(des_img, prev_point, point, color, thickness)
                prev_point = point

            if len(point_list) == 4:
                cv2.line(des_img, point_list[-1], point_list[0], color, thickness)

                point_list = sort_points(point_list)

        cv2.imshow('img', des_img)


# 마우스 이벤트 콜백 함수


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
new_point_list = []
src_img = cv2.imread('input4.png')

width = 640
height = 640
color = (0,255,255)
thickness = 5
drawing = False
des_img = None

cv2.imshow('img', src_img)
cv2.setMouseCallback('img', mouse_handler)
cv2.waitKey(0)
cv2.destroyAllWindows()


homography()
cv2.waitKey(0)

img = img_homo

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


#
for i in range(len(sorted_x)):
    if np.abs(sorted_x[i + 1][0][0] - sorted_x[i][0][0]) >= 50:
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
for point in sorted_xy:
    pt = (point[0][0], point[0][1])
    length = 5
    cv2.line(img, (pt[0], pt[1]-length), (pt[0], pt[1]+length), (255, 0, 0), 5) # vertical line
    cv2.line(img, (pt[0]-length, pt[1]), (pt[0]+length, pt[1]), (255, 0, 0), 5)

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

# 깨진 타일 가로,세로 구하기
if aver_y[0] > 10 :
    u_tile = [(aver_w,aver_y[0])] * (num_vertical-1)                  # 위
else :
    u_tile = []

if (height-aver_y[-1]) > 10 :
    d_tile = [(aver_w,height-aver_y[-1])] * (num_vertical-1)          # 아래
else :
    d_tile = []

if aver_x[0] > 10 :
    l_tile = [(aver_x[0],aver_h)] * (num_horizon -1)                  # 좌
else :
    l_tile = []

if (width-aver_x[-1]) > 10 :
    r_tile = [(width - aver_x[-1],aver_h)] * (num_horizon -1)         # 우
else :
    r_tile = []

if aver_y[0] > 10 and aver_x[0] > 10 :
    lu_tile = [(aver_x[0],aver_y[0])]               # 좌상 모서리
else :
    lu_tile = []

if aver_y[0] > 10 and (width-aver_x[-1]) > 10 :
    ru_tile = [(width - aver_x[-1],aver_y[0])]                        # 우상 모서리
else :
    ru_tile = []

if (height-aver_y[-1]) > 10 and aver_x[0] > 10 :
    ld_tile = [(aver_x[0],height-aver_y[-1])]                         # 좌하 모서리
else :
    ld_tile = []

if (height-aver_y[-1]) > 10 and width - aver_x[-1] > 10 :
    rd_tile = [(width - aver_x[-1],height-aver_y[-1])]               # 우하 모서리
else :
    rd_tile = []
print('윗줄 깨진 타일',u_tile)
print('아랫줄 깨진 타일',d_tile)
print('왼쪽 깨진 타일',l_tile)
print('오른쪽 깨진 타일',r_tile)
print('좌상 모서리',lu_tile)
print('우상 모서리',ru_tile)
print('좌하 모서리',ld_tile)
print('우하 모서리',rd_tile)

b_tile = u_tile + d_tile + l_tile + r_tile + lu_tile + ru_tile + ld_tile + rd_tile

# Rectpack 패커 객체 생성 및 설정
bins = [(aver_w, aver_h)]*len(b_tile)
print("너비, 높이, 면적:", aver_w, aver_h, aver_w * aver_h)
rectangles = b_tile

packer = newPacker(mode=PackingMode.Offline, rotation=True)

# Add the rectangles to packing queue
for r in rectangles:
    packer.add_rect(*r)

# Add the bins where the rectangles will be placed
for b in bins:
    packer.add_bin(*b)

# Start packing
packer.pack()

# Print the required number of bins
print(len(packer))

# Print the packing results
for i, b in enumerate(packer):
    print(f"Bin {i+1}:")
    for r in b:
        print(f"\t{(r.width, r.height)}\t{(r.x, r.y)}")


n_subplots = len(packer)
nrows = (n_subplots - 1) // 5 + 1  # 한 열에 5개 subplot이 들어가므로
ncols = min(n_subplots, 5)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3*nrows))

for i, b in enumerate(packer):
    row_idx, col_idx = divmod(i, 5)
    if nrows == 1:
        ax = axs[col_idx]
    else:
        ax = axs[row_idx, col_idx]

    # Plot the bin
    bin_rect = Rectangle((0, 0), bins[i][0], bins[i][1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(bin_rect)

    # Plot the rectangles
    for r in b:
        rect = Rectangle((r.x, r.y), r.width, r.height, linewidth=1, edgecolor='k', facecolor='g', alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, bins[i][0])
    ax.set_ylim(0, bins[i][1])
    ax.set_aspect('equal')
    ax.set_title(f"Bin {i+1}")

# Adjust the spacing between subplots and show the plot
plt.tight_layout()

print('께진 타일 수 :',len(rectangles))
print("깨진 타일로 패킹하여 만든 온장 수 :",len(packer))
print("깨지지 않은 타일 수 : ",whole_tile)


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()