import cv2
import numpy as np

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global pt_list, count, img

    if event == cv2.EVENT_LBUTTONDOWN:
        if count < 4:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            pt_list.append([x, y])
            count += 1
            if count > 1:
                cv2.line(img, (x, y), tuple(pt_list[-2]), (255, 0, 0), thickness=2)
            cv2.imshow('Original Image', img)

from collections import defaultdict

#function to do the segmentation
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


# 이미지 로드
img = cv2.imread('Base01.jpg')

# 마우스 이벤트 처리를 위한 변수 초기화
pt_list = []
count = 0

# 이미지 표시 및 마우스 이벤트 처리
cv2.imshow('Original Image', img)
cv2.setMouseCallback('Original Image', mouse_callback)

# 마우스 이벤트 처리 완료 후, 호모그래피 계산
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break
    elif count == 4:
        pts1 = np.float32(pt_list)
        pts2 = np.float32([[0, 0], [0, 1000], [1000, 1000], [1000, 0]])
        h, _ = cv2.findHomography(pts1, pts2)
        img_out = cv2.warpPerspective(img, h, (1000, 1000))
        homography = img_out

        # 이미지를 그레이스케일로 변환합니다.
        gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        # 캐니 엣지 검출기를 사용하여 엣지를 검출합니다.
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 허프 변환을 사용하여 직선을 검출합니다.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        # 검출된 직선을 그립니다.
        hough_image = np.zeros_like(img_out)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #trial point
        segmented = segment_by_angle_kmeans(lines)
        intersections = segmented_intersections(segmented)


        numpy_horizontal = np.hstack((homography, hough_image))
        cv2.imshow('Intersection Points', numpy_horizontal)
# 결과 이미지를 출력합니다.
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


