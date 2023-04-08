import cv2
import numpy as np

# 이미지 파일 읽기
img = cv2.imread('saved_image.jpg')

# 이미지 크기 확인
height, width, channels = img.shape

# 마우스 클릭 위치 저장
clicks = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        clicks.append((x, y))


# 이미지 창 열기 및 마우스 콜백 등록
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    # 이미지 표시
    cv2.imshow('image', img)
    cv2.waitKey(1)

    # 마우스 클릭이 4개 이상 되면 루프 종료
    if len(clicks) == 4:
        break

# 클릭된 점의 좌표를 정렬하여 면적 계산에 사용
clicks = sorted(clicks)

# 클릭된 영역의 넓이 계산
area = cv2.contourArea(np.array(clicks))

# 결과 출력
print("선택한 영역의 넓이: ", area)

# 이미지 창 종료
cv2.destroyAllWindows()
