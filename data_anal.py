import cv2
import mediapipe as mp
import mediapipe.tasks.python.components.containers
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product

# 1. 미디어파이프를 사용하여 비디오에서 스켈레톤 구조 추출하기
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('./videos/Standard.mp4')  # 비디오 파일 경로
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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            pose_data.append([landmark.x for landmark in landmarks])



cap.release()
pose_landmark_names = [
    'nose',
    'left_eye_inner',
    'left_eye',
    'left_eye_outer',
    'right_eye_inner',
    'right_eye',
    'right_eye_outer',
    'left_ear',
    'right_ear',
    'mouth_left',
    'mouth_right',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_pinky',
    'right_pinky',
    'left_index',
    'right_index',
    'left_thumb',
    'right_thumb',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
    'left_heel',
    'right_heel',
    'left_foot_index',
    'right_foot_index'
]

# 2. 시계열 데이터로 변환
pose_df = pd.DataFrame(pose_data)
pose_df.columns = pose_landmark_names
pose_df.to_csv("./")
print(pose_df)

pose_ts = pose_df.mean(axis=1)  # 관절의 평균 위치를 사용. 필요한 경우 다른 통계치를 사용할 수 있음

# 3. 데이터 전처리 (정상화)
diff = pose_ts.diff().dropna()
result = adfuller(diff)
stationary_data = diff if result[1] < 0.05 else pose_ts
# 4. SARIMA 모델 학습 및 최적 매개변수 찾기
p = d = q = P = D = Q = range(0, 2)
s = [fps]  # 비디오의 초당 프레임 수를 계절성 주기로 설정
pdq = list(product(p, d, q))
seasonal_pdq = list(product(P, D, Q, s))

best_aic = float('inf')
best_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(stationary_data,
                            order=param,
                            seasonal_order=seasonal_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, seasonal_param)
        except Exception as e:
            continue

# 5. 최적 매개변수를 사용하여 SARIMA 모델 학습
optimal_model = SARIMAX(stationary_data,
                        order=best_params[0],
                        seasonal_order=best_params[1],
                        enforce_stationarity=False,
                        enforce_invertibility=False)
optimal_results = optimal_model.fit()

# 6. 주기성 및 특징 분석
import matplotlib.pyplot as plt

# 주기성 확인 (계절성 성분)
seasonal_component = optimal_results.seasonal_components
plt.plot(seasonal_component)
plt.title('Seasonal Component')
plt.show()

# 추세 확인 (추세 성분)
trend_component = optimal_results.trend_components
plt.plot(trend_component)
plt.title('Trend Component')
plt.show()

# 예측
forecast_steps = 100  # 예측할 프레임 수
forecast = optimal_results.get_forecast(steps=forecast_steps)
forecast_confidence_intervals = forecast.conf_int()

# 예측 결과 시각화
plt.plot(stationary_data, label='Observed')
plt.plot(forecast.predicted_mean, label='Forecast', color='r')
plt.fill_between(forecast_confidence_intervals.index,
                 forecast_confidence_intervals.iloc[:, 0],
                 forecast_confidence_intervals.iloc[:, 1], color='pink')
plt.xlabel('Frame')
plt.ylabel('Pose Data')
plt.legend()
plt.show()

