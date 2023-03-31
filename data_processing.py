import pandas as pd
import numpy as np

data = pd.read_csv('./data/pose.csv')
# columns = ["left_shoulder_x","left_shoulder_y","right_shoulder_x","right_shoulder_y","left_elbow_x","left_elbow_y","right_elbow_x","right_elbow_y",
#            "left_wrist_x","left_wrist_y","right_wrist_x","right_wrist_y"]
data['right']