import cv2, dlib
from imutils import face_utils, resize
import numpy as np

orange_img = cv2.imread('orange.jpg') # 오렌지 이미지 준비한 것을 opencv의 imread로 읽어서 orange_img에 저장 
orange_img = cv2.resize(orange_img, dsize=(512,512)) # 가로 512 세로 512로 resize

detector = dlib.get_frontal_face_detector() # dlib 안에 있는 얼굴 영역 탐지 관련 메소드를 초기화 해주고 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # ** 68개 점의 랜드 마크를 탐지 **

cap = cv2.VideoCapture('video.mp4') # 0이면 본인의 웹캠이 켜지고 영상 파일 이름을 넣으면 영상 파일이 켜진다. 

while cap.isOpened():
    ret, img = cap.read() # read를 통해 img를 읽어오고 

    if not ret: # 보낼 프레임이 없다면 종료 
        break

    faces = detector(img) # 얼굴 영역을 인식해줌, 이때 faces에 얼굴 영역 좌표가 담김

    result = orange_img.copy() #Orange 이미지를 복사한 것 

    if len(faces) > 0: # 얼굴이 1개 이상이면, 즉 프레임이 여러개이므로 detect된 얼굴도 여러 개이겠지
        face = faces[0] # 그중에서 1개(첫 번째)의 얼굴만을 원한다. 

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy() # img에서 해당 face만 crop(잘라내기)해서 face_img에 저장

        # 랜드마크의 68점을 찾는다
        shape = predictor(img, face) 
        shape = face_utils.shape_to_np(shape) # 원래 shape는 dlib형태인데 이를 numpy 형태로 바꿔준다. 

        for p in shape:
            cv2.circle(face_img, center=(p[0]-x1, p[1]-y1), radius=2, color=255, thickness=-1)

        # ** 눈과 입을 잘라서 오렌지 파일에 붙이는 코드 **
        # eyes
        le_x1 = shape[36, 0] # **왼쪽 눈을 자르기 위해 x좌표 36번 39번, y좌표 37번 41번 인덱스가 필요**
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.18) # 너무 가깝게 자르면 안 되니까 margin을 준다 

        re_x1 = shape[42, 0] # 오른쪽 눈
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18)

        # 여기서 왼쪽 눈과 오른쪽 눈에 margin을 준 값을 crop한다. 
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy() 
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()

        left_eye_img = resize(left_eye_img, width=100) # 가로를 100으로 resize
        right_eye_img = resize(right_eye_img, width=100) 

        # ** opencv의 seamlessClone **
        result = cv2.seamlessClone( # 왼쪽 눈 합성  
            left_eye_img,
            result, #result(오렌지 이미지를 복사한 것)에 합성 하는 것 
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (100, 200),
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone( # 오른쪽 눈 합성 
            right_eye_img,
            result,
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (250, 200),
            cv2.MIXED_CLONE
        )

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        mouth_img = resize(mouth_img, width=250)

        result = cv2.seamlessClone( # 입 합성 
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 320), # 합성할 때 좌표를 지정해주는 부분 
            cv2.MIXED_CLONE
        )

        cv2.imshow('left', left_eye_img)
        cv2.imshow('right', right_eye_img)
        cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)

        cv2.imshow('result', result)

    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
