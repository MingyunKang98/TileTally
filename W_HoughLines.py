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
            cv2.imshow('image', img)


# 이미지 로드
img = cv2.imread('tile8.jpg')

# 마우스 이벤트 처리를 위한 변수 초기화
pt_list = []
count = 0

# 이미지 표시 및 마우스 이벤트 처리
cv2.imshow('image', img)
cv2.setMouseCallback('image', mouse_callback)

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
        # img_out = cv2.rotate(img_out, cv2.ROTATE_90_CLOCKWISE)  # 이미지 90도 회전

        # 그레이 스케일 변환
        gray_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 필터 적용
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # 캐니 엣지 검출기 적용
        edges = cv2.Canny(blur_img, 50, 100, apertureSize=3)

        # 허프 변환 수행
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=70)

        # RANSAC으로 수평과 수직 직선 추출
        vertical_lines = []
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:
                vertical_lines.append(line)
            elif abs(y2 - y1) < 5:
                horizontal_lines.append(line)

        # 수평과 수직 직선 병합
        merged_lines = []
        for v_line in vertical_lines:
            v_x1, v_y1, v_x2, v_y2 = v_line
            for h_line in horizontal_lines:
                h_x1, h_y1, h_x2, h_y2 = h_line
                # 수직선의 기울기가 수평선과 90도로 수직이라면
                if abs(v_x1 - v_x2) < 10 and abs(h_y1 - h_y2) < 10:
                    x, y = v_x1, h_y1
                    merged_lines.append([x, y])
                    # 직선 그리기
                    cv2.line(line_img, (x - 20, y), (x + 20, y), (0, 0, 255), 2)
                    cv2.line(line_img, (x, y - 20), (x, y + 20), (0, 0, 255), 2)

        # 이미지 합치기 단계
        result_img = np.concatenate((img_out, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), line_img), axis=1)
        cv2.imshow('Result', result_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

