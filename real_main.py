import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy.signal import periodogram

def calculate_length(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_angle(p1: tuple[int, int], p2: tuple[int, int], p3: tuple[int, int]):
    a = calculate_length(p1, p2)
    b = calculate_length(p2, p3)
    c = calculate_length(p3, p1)
    return math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))


def get_landmark_coordinates(landmark, width, height) -> tuple[float, float]:
    return (landmark.x * width), (landmark.y * height)


def process_video(video_path: str, landmark_angle: dict):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    pose_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            height, width, _ = image.shape
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                tmp = []
                for angle in landmark_angle.values():
                    cal_angle = calculate_angle(
                        get_landmark_coordinates(landmarks[angle[0]], width, height),
                        get_landmark_coordinates(landmarks[angle[1]], width, height),
                        get_landmark_coordinates(landmarks[angle[2]], width, height),
                    )
                    tmp.append(cal_angle)
                pose_data.append(tmp)
    cap.release()
    return pose_data
landmark_angle = {
        "left_elbow_angle": (11, 13, 15),
        "right_elbow_angle": (12, 14, 16),
        "left_wrist_angle": (13, 15, 17),
        "right_wrist_angle": (14, 16, 18),
        "left_shoulder_angle": (23, 11, 13),
        "right_shoulder_angle": (24, 12, 14),
        "hip_angle": (23, 24, 26),
        "waist_angle": (23, 11, 24),
        "left_knee_angle": (23, 25, 27),
        "right_knee_angle": (24, 26, 28),
        "left_ankle_angle": (25, 27, 29),
        "right_ankle_angle": (26, 28, 30),
    }

def extractData(video_path):    
    pose_data = process_video(video_path, landmark_angle)
    pose_df = pd.DataFrame(pose_data)
    pose_df.columns = landmark_angle.keys()
    pose_df.to_csv("./data/pose3.csv", index=False)
    return pose_df

def processData(pose_df):
    # pose_df = pd.read_csv("./data/pose3.csv")
    landmark_columns = pose_df.columns

    power_df = pd.DataFrame(columns=['landmark', 'period', 'power'])

    for column_name in landmark_columns:
        data = pose_df[column_name].values
        period, power = find_dominant_period(data)
        power_df = power_df.append({'landmark': column_name, 'period': period, 'power': power.max()}, ignore_index=True)

    threshold = power_df['power'].mean() + power_df['power'].std() * 1/3
    power_df = power_df[power_df['power'] > threshold]

    periodic_landmark_ranges = []

    for column_name, period in power_df[['landmark', 'period']].values:
        landmark_range = np.array([])

        for i in range(0, len(pose_df), int(period)):
            landmark_range = np.append(landmark_range, [pose_df[column_name][i:i + int(period)].values.max(), pose_df[column_name][i:i + int(period)].values.min()])

        landmark_range = landmark_range.reshape(-1, 2)
        landmark_range_mean = (int(landmark_range[:, 0].mean()), int(landmark_range[:, 1].mean()))
        periodic_landmark_ranges.append((column_name, landmark_range_mean))

    periodic_landmark_ranges = pd.DataFrame(periodic_landmark_ranges, columns=['landmark', 'range'])
    periodic_landmark_ranges.to_csv('./data/periodic_landmark_ranges.csv', index=False)
    return periodic_landmark_ranges

# 주기성을 판단하는 함수
def find_dominant_period(data):
    # periodogram: 주파수와 주파수에 해당하는 파워를 계산합니다.(파워: 특정 주파수의 세기)
    freqs, power = periodogram(data, fs = 0.5)
    dominant_freq = freqs[np.argmax(power)]
    dominant_period = 1 / dominant_freq
    return dominant_period, power

def showVideo(video_path, rangeData):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    ex_range = rangeData.values.tolist()
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("비디오를 읽을 수 없습니다.")
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            height, width, _ = image.shape
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                for i, (name, angle_range) in enumerate(ex_range):
                    (MAX_ANGLE, MIN_ANGLE) = angle_range
                    MAX_ANGLE = MAX_ANGLE + (MAX_ANGLE - MIN_ANGLE) * 0.05
                    MIN_ANGLE = MIN_ANGLE - (MAX_ANGLE - MIN_ANGLE) * 0.05
                    a, b, c = landmark_angle[name]
                    x, y = get_landmark_coordinates(landmarks[b], width, height)
                    angle = calculate_angle(get_landmark_coordinates(landmarks[a], width, height),
                            get_landmark_coordinates(landmarks[b], width, height),
                            get_landmark_coordinates(landmarks[c], width, height))
                    color = (0, 255, 0) if MIN_ANGLE <= angle <= MAX_ANGLE else (0, 0, 255)
                    cv2.putText(frame, f"{name}: {int(angle)}", (50, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    if MIN_ANGLE > angle or angle > MAX_ANGLE:
                        cv2.putText(frame, str(int(angle)), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
def main():
    video_path = "./videos/pushup.mp4"
    pose_df = extractData(video_path)
    rangeData = processData(pose_df)
    showVideo(video_path, rangeData)
    
if __name__ == "__main__":
    main()