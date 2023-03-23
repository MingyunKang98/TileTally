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

        numpy_horizontal = np.hstack((homography, hough_image))
        cv2.imshow('Intersection Points', numpy_horizontal)
# 결과 이미지를 출력합니다.
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



