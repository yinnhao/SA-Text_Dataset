import cv2
import numpy as np

# 0으로 채워진 이미지(array) 생성
image = np.zeros((150,150,3),np.uint8)

cv2.imwrite('./img1.jpg', image)

# 다각형의 좌표를 순서대로 입력, 입력 순서대로 선을 그리게 된다
pts1 = np.array([[20,20],[130,70],[90,90],[80,130]],dtype=np.int32)

# cv2의 polylines로 다각형 그리기
# cv2.polylines(원본그림, 좌표리스트, 마지막점과 첫점 연결 여부, 선색, 선 굵기)
cv2.polylines(image, [pts1], False, (255,255,255), 4)
cv2.imwrite('./img2.jpg', image)

breakpoint()