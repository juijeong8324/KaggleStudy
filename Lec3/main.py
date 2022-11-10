import cv2
import numpy as np
import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Input video path')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0) 
# 비디오 파일을 읽는다. 비디오가 없다면 0으로 설정하여 웹캠이 켜진다. 

time.sleep(3)
# 3초간 정지 = 카메라를 켜지는데 시간이 걸리기 때문

# Grap background image from first part of the video **비디오 앞 부분에 사람이 나오지 않은 배경이 꼭 필요(중요)**
for i in range(60):
  ret, background = cap.read()

# 동영상의 결과값을 기록하기 위한 코드! 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('videos/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))
out2 = cv2.VideoWriter('videos/original.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))

while(cap.isOpened()):
  ret, img = cap.read() # 한 프레임씩 읽어온다! 원본 이미지가 img에 저장됨
  if not ret:
    break
  
  # Convert the color space from BGR to HSV
  # cv2.cvtColor() : 컬러 시스템을 변경한다. 
  # **원본 이미지를 BGR에서 HSV로 바꾼다.** 
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Generate mask to detect red color
  # 빨간색의 범위는 0~10 170~180이라 두 개로 나누어 마스크 생성 후 더함! 
  # 0~10까지 빨간색
  lower_red = np.array([0, 120, 70]) # 채도는 120 밝기는 70
  upper_red = np.array([10, 255, 255]) # 채도는 255 밝기는 255
  mask1 = cv2.inRange(hsv, lower_red, upper_red) # **cv2.inRange() : 범위 안에 해당하는 값들로 마스크를 생성**

  # 170~180까지 빨간색
  lower_red = np.array([170, 120, 70])
  upper_red = np.array([180, 255, 255])
  mask2 = cv2.inRange(hsv, lower_red, upper_red)

  mask1 = mask1 + mask2 # 빨간색을 다 mask한 마스크가 탄생

# 검정색을 가리고 싶을 떄 
  # lower_black = np.array([0, 0, 0])
  # upper_black = np.array([255, 255, 80])
  # mask1 = cv2.inRange(hsv, lower_black, upper_black)

  '''
  # Refining the mask corresponding to the detected red color
  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  '''

  # Remove noise **마스크를 색만 뽑으면 노이즈가 발생해서 정제해주는 함수임**
  mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2) #노이즈 없애고
  mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1) # 픽셀 크기 늘리기 
  mask_bg = cv2.bitwise_not(mask_cloak)

  cv2.imshow('mask_cloak', mask_cloak)

  # Generate the final output
  # bitwise_and() : 두개의 행렬이 0이 아닌 것만 통과됨 즉 마스크 영역만 남음(And 연산)
  res1 = cv2.bitwise_and(background, background, mask=mask_cloak) # background에서 mask만 남고 
  res2 = cv2.bitwise_and(img, img, mask=mask_bg) # 캠으로 들어오는 이미지에서 mask가 안 된 부분만 남고 
  result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0) 
  # cv2.addWeighted() 두 개의 이미지를 합친다 

  cv2.imshow('res1', res1)

  # cv2.imshow('ori', img)
  cv2.imshow('result', result)
  out.write(result)
  out2.write(img)

  if cv2.waitKey(1) == ord('q'):
    break

out.release()
out2.release()
cap.release()