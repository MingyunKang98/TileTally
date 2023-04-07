import cv2
import numpy as np
from keras.models import load_model

# 딥러닝 모델 로드
model = load_model('rectangle_detection.h5')

# 이미지 읽기
img = cv2.imread('Base01.jpg')

# 이미지 전처리
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)

# 직사각형 검출
predictions = model.predict(img)
if predictions > 0.5:
    # 직사각형이 감지된 경우
    # 직사각형 좌표 계산
    x1 = int(predictions[1][0])
    y1 = int(predictions[2][0])
    x2 = int(predictions[3][0])
    y2 = int(predictions[4][0])
    # 직사각형을 빨간색 사각형으로 표시
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
# 직사각형이 감지되지 않은 경우
# 해당하는 코드 작성
    print('No rectangle found in the image.')
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
