import cv2
import mediapipe as mp
import numpy as np
import winsound
import threading
from tkinter.filedialog import askopenfilename
import tkinter as tk

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Tạo root ẩn để dùng file dialog mà không hiện cửa sổ Tkinter
root = tk.Tk()
root.withdraw()

video_path = askopenfilename(title="Chọn file video hoặc nhấn Cancel để dùng webcam", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
if video_path:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)

def play_alarm():
    winsound.PlaySound('alarm.wav', winsound.SND_FILENAME)

def is_fallen(landmarks):
    try:
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                         landmarks[mp_pose.PoseLandmark.NOSE.value].y])

        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2

        body_vec = mid_shoulder - mid_hip
        body_angle = np.arctan2(body_vec[1], body_vec[0]) * 180 / np.pi

        if abs(body_angle) > 45 and nose[1] > mid_hip[1]:
            return True
        return False
    except:
        return False

alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    fallen = False
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        fallen = is_fallen(landmarks)
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if fallen:
        cv2.putText(frame, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        if not alarm_playing:
            alarm_playing = True
            threading.Thread(target=play_alarm, daemon=True).start()
    else:
        alarm_playing = False

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(30 if video_path else 1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 