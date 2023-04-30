import cv2
import mediapipe as mp
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
video_path = "./videos/shoulder.mp4"

columns = ["left_shoulder_x","left_shoulder_y","right_shoulder_x","right_shoulder_y","left_elbow_x","left_elbow_y","right_elbow_x","right_elbow_y",
           "left_wrist_x","left_wrist_y","right_wrist_x","right_wrist_y"]
pose_list = pd.DataFrame(columns=columns)


cap = cv2.VideoCapture(video_path)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      break
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
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    pose_landmarks = results.pose_landmarks
    if pose_landmarks is not None:
        left_shoulder_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width
        left_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height
        right_shoulder_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * width
        right_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * height
        left_elbow_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width
        left_elbow_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height
        right_elbow_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * width
        right_elbow_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * height
        left_wrist_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width
        left_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height
        right_wrist_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width
        right_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * height
        
        pose_list.loc[len(pose_list)] = [left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y,
                                         left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
                                         left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y]


    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
print(pose_list)
pd.DataFrame.to_csv(pose_list,"./data/pose.csv")
