import math
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 미디어 파이프의 그리기 유틸리티 및 스타일을 가져옵니다.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# 미디어 파이프의 포즈 솔루션을 가져옵니다.
mp_pose = mp.solutions.pose
def calculate_length(p1 :tuple[int,int] , p2:tuple[int,int]) -> float:
    """
    두 점 사이의 거리를 계산합니다. (유클라디안 거리)
    :param p1: 첫 번째 점 (x, y)
    :param p2: 두 번째 점 (x, y)
    :return: 두 점 사이의 거리
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_angle(p1:tuple[int,int], p2:tuple[int,int], p3:tuple[int,int]):
    """
    삼각형의 세 점 사이의 각도를 계산합니다.
    :param p1: 첫 번째 점 (x, y)
    :param p2: 두 번째 점 (x, y)
    :param p3: 세 번째 점 (x, y)
    :return: 삼각형의 세 점 사이의 각도 ∠p1p2p3
    """
    # 삼각형의 세 점 사이의 각도를 계산합니다.
    a = calculate_length(p1, p2)
    b = calculate_length(p2, p3)
    c = calculate_length(p3, p1)
    # 코사인 법칙을 사용하여 각도를 계산합니다.
    # ∠abc = arccos((a² + b² - c²) / 2ab)
    return math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))

def get_landmark_coordinates(landmark, width, height) -> tuple[int,int]:
    """
    랜드마크의 좌표를 계산합니다.
    :param landmark: 랜드마크
    :param width: 이미지의 너비
    :param height: 이미지의 높이
    :return: 랜드마크의 좌표 (x, y)
    """
    return int(landmark.x * width), int(landmark.y * height)

def process_frame(image, pose) -> tuple[np.ndarray, mp_pose.PoseLandmark, int, int]:
    """
    비디오의 단일 프레임을 처리합니다.
    :param image: 처리할 이미지
    :param pose: 포즈 모델
    :return: 포즈 랜드마크가 그려진 이미지
    """
    height, width, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    return image, results.pose_landmarks, width, height

randmark_angle = {
    'left_elbow_angle':(11,13,15),
    'right_elbow_angle':(12,14,16),
    'left_wrist_angle':(13,15,17),
    'right_wrist_angle':(14,16,18),
    'left_shoulder_angle':(23,11,13),
    'right_shoulder_angle':(24,12,14),
    'hip_angle':(23,24,26),
    'waist_angle':(23,11,24),
    'left_knee_angle':(23,25,27),
    'right_knee_angle':(24,26,28),
    'left_ankle_angle':(25,27,29),
    'right_ankle_angle':(26,28,30)
}

# csv 파일 불러오기
ex_range = pd.read_csv('./data/periodic_landmark_ranges.csv')
VIDEO_PATH = './vidios/pushup.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)

count = 0

is_started = False
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("비디오를 읽을 수 없습니다.")
            break
        image, pose_landmarks, width, height = process_frame(image, pose)
        
        if pose_landmarks is not None:
            # 랜드마크의 좌표를 계산하고 딕셔너리에 저장합니다.
            landmarks = {name: get_landmark_coordinates(pose_landmarks.landmark[landmark.value], width, height)
                            for name, landmark in mp_pose.PoseLandmark.__members__.items()}
            for name, (MAX_ANGLE, MIN_ANGLE) in ex_range.items():
                a,b,c = randmark_angle[name]
                angle = calculate_angle(landmarks[a],landmarks[b],landmarks[c])
                # 설정된 각도 범위에 따라 색상을 결정합니다. (초록색: 정상, 빨간색: 비정상)
                right_color = (0, 255, 0) if MIN_ANGLE <= right_angle <= MAX_ANGLE else (0, 0, 255)
                cv2.putText(image, str(int(angle)), landmarks[b], cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            # # 각도를 통해 운동 횟수를 카운트합니다.
            # if left_angle < START_ANGLE and right_angle < START_ANGLE:
            #     is_started = True
            # if is_started and left_angle > END_ANGLE and right_angle > END_ANGLE:
            #     is_started = False
            #     count += 1

            # # 카운트를 이미지에 표시합니다.
            # cv2.putText(image, f"count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        # 결과 이미지를 표시합니다.    
        cv2.imshow('MediaPipe Pose', image)
        # ESC 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == 27:
            break