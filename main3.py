import math
import cv2
import mediapipe as mp

# 미디어 파이프의 그리기 유틸리티 및 스타일을 가져옵니다.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# 미디어 파이프의 포즈 솔루션을 가져옵니다.
mp_pose = mp.solutions.pose

# 입력 비디오 파일 경로 및 각도 범위를 설정합니다.
VIDEO_PATH = "./videos/shoulder.mp4"
MIN_ANGLE = 50
MAX_ANGLE = 150

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

def get_landmark_coordinates(landmark, width, height):
    """
    랜드마크의 좌표를 계산합니다.
    :param landmark: 랜드마크
    :param width: 이미지의 너비
    :param height: 이미지의 높이
    :return: 랜드마크의 좌표 (x, y)
    """
    return int(landmark.x * width), int(landmark.y * height)

def process_frame(image, pose):
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

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
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
                # 왼쪽과 오른쪽 팔의 각도를 계산합니다.
                left_angle = calculate_angle(landmarks['LEFT_SHOULDER'], landmarks['LEFT_ELBOW'], landmarks['LEFT_WRIST'])
                right_angle = calculate_angle(landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_ELBOW'], landmarks['RIGHT_WRIST'])

                # 설정된 각도 범위에 따라 색상을 결정합니다. (초록색: 정상, 빨간색: 비정상)
                left_color = (0, 255, 0) if MIN_ANGLE <= left_angle <= MAX_ANGLE else (0, 0, 255)
                right_color = (0, 255, 0) if MIN_ANGLE <= right_angle <= MAX_ANGLE else (0, 0, 255)

                # 이미지에 왼쪽과 오른쪽 팔의 각도를 표시합니다.
                cv2.putText(image, str(int(left_angle)), landmarks['LEFT_ELBOW'], cv2.FONT_HERSHEY_SIMPLEX, 1.0, left_color, 2, cv2.LINE_AA)
                cv2.putText(image, str(int(right_angle)), landmarks['RIGHT_ELBOW'], cv2.FONT_HERSHEY_SIMPLEX, 1.0, right_color, 2, cv2.LINE_AA)

            # 결과 이미지를 표시합니다.
            cv2.imshow('MediaPipe Pose', image)
            # ESC 키를 누르면 종료합니다.
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
if __name__ == "__main__":
    main()
