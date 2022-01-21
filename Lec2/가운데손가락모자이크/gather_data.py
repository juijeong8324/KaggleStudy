import cv2 # 웹캠 제어 및 ML 사용 
import mediapipe as mp # 손 인식을 할 것
import numpy as np

max_num_hands = 1 # 손은 최대 1개만 인식
gesture = { # 제스처의 클래스를 정의하는 부분에서 11번:fy 클래스 만들기
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
} 

# MediaPipe hands model
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

 # 손가락 detection 모듈을 초기화
hands = mp_hands.Hands(  
    max_num_hands=max_num_hands, # 최대 몇 개의 손을 인식? 
    min_detection_confidence=0.5, # 0.5로 해두는 게 좋다!  
    min_tracking_confidence=0.5)  

# 제스처 인식 모델 
file = np.genfromtxt('gesture_train.csv', delimiter=',') # **각 제스처들의 라벨과 각도가 저장되어 있음, 정확도를 높이고 싶으면 데이터를 추가해보자!** 
print(file.shape)

cap = cv2.VideoCapture(0) 

# **화면을 클릭했을 때만 데이터를 저장** 
# 즉 클릭했을 때 현재 각도 data를 원본 file에 추가하도록 한다. 
def click(event, x, y, flags, param): # clik 핸들러
    global data, file
    if event == cv2.EVENT_LBUTTONDOWN:
        file = np.vstack((file, data)) # numpy의 vstack을 사용해서 이어붙여준다.  
        print(file.shape) 
cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

while cap.isOpened(): # 웹캠에서 한 프레임씩 이미지를 읽어옴
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            data = np.append(data, 11) # 각도 데이터의 마지막에 정답 라벨인 숫자 11을 추가

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'): #q를 눌러서 while 루프 빠져나오고 데이터 수집 종료 
        break

np.savetxt('gesture_train_fy.csv', file, delimiter=",") # 수집한 데이터를 txt 파일로 저장 