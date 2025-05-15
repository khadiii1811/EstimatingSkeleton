import cv2
import numpy as np
import threading
import winsound
import os
import argparse
from tkinter import filedialog, Toplevel, messagebox

# Utility functions for processing, sound, etc.

def denoise(frame):
    """Apply denoising filters to improve image quality"""
    frame = cv2.medianBlur(frame, 11)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    return frame

# Global variable to control alarm playback
alarm_playing = False

def play_alarm():
    """Play alarm sound file"""
    try:
        winsound.PlaySound('alarm.wav', winsound.SND_FILENAME)
    except Exception as e:
        print(f"Error playing alarm: {str(e)}")
    finally:
        global alarm_playing
        alarm_playing = False

def play_alarm_thread():
    """Start alarm in separate thread to avoid blocking"""
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        threading.Thread(target=play_alarm, daemon=True).start()

def load_cnn_model():
    """Load and configure CNN-based pose estimation model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    args = parser.parse_args([])
    
    # Define body parts mapping
    BODY_PARTS = { 
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 
    }

    # Define pose pairs for drawing
    POSE_PAIRS = [ 
        ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] 
    ]
    
    # Load model from file
    try:
        net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
        print("CNN model loaded successfully")
        return net, args, BODY_PARTS, POSE_PAIRS
    except Exception as e:
        print(f"Error loading CNN model: {str(e)}")
        return None, args, BODY_PARTS, POSE_PAIRS

def select_video_file(title="Select video file"):
    """Open file dialog to select a video file"""
    video_path = filedialog.askopenfilename(
        title=title, 
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    return video_path

def create_choice_dialog(root, title, message, options, commands):
    """Create a dialog with multiple choices"""
    dialog = Toplevel(root)
    dialog.title(title)
    dialog.geometry("300x200")
    dialog.configure(bg='#313244')
    dialog.resizable(False, False)
    
    # Center the window
    dialog.geometry("+%d+%d" % (root.winfo_rootx() + 50, 
                                root.winfo_rooty() + 50))
    
    # Add message label
    from tkinter import Label, Button, X
    Label(dialog, text=message, 
          font=('Helvetica', 12), bg='#313244', fg='#CDD6F4').pack(pady=10)
    
    # Add option buttons
    btn_font = ('Helvetica', 12)
    
    for i, option in enumerate(options):
        Button(dialog, text=option, font=btn_font, bg='#89B4FA', fg='#CDD6F4',
              command=lambda idx=i: [dialog.destroy(), commands[idx]()]).pack(fill=X, padx=20, pady=5)
    
    # Add cancel button
    Button(dialog, text="Cancel", font=btn_font, bg='#F38BA8', fg='#CDD6F4',
          command=dialog.destroy).pack(fill=X, padx=20, pady=5)
    
    # Focus on dialog
    dialog.transient(root)
    dialog.grab_set()
    
    return dialog

def export_data_to_file(data, file_path, file_type):
    """Export data to a file with error handling"""
    try:
        if file_type == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
            return True, "Data exported successfully"
        else:
            return False, f"Unsupported file type: {file_type}"
    except Exception as e:
        return False, f"Error exporting data: {str(e)}" 