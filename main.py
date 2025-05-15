from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import threading
import signal
import sys
import time

# Import modular components
from detectors import PoseDetector
from video_processor import (
    update_preview_image, run_video_processing, 
    process_combination, process_face_detection, process_hand_tracking, 
    process_face_mesh, process_pose_estimation, process_fall_detection
)
from evaluation import (
    evaluation_data, reset_evaluation_data, session_history,
    show_evaluation_report, show_performance_comparison
)
from ui_components import create_ui
from utils import play_alarm_thread, select_video_file

def main():
    # Create a window
    root = Tk()
    root.title('Human Skeleton Estimating - Team 4')
    root.configure(bg='#1E1E2E')
    root.geometry('1280x800')

    # Setup custom fonts and colors
    title_font = ('Helvetica', 22, 'bold')
    heading_font = ('Helvetica', 22, 'bold')
    btn_font = ('Helvetica', 12)
    label_font = ('Helvetica', 12)

    # Color palette - dark modern theme
    bg_color = '#1E1E2E'  # Darker background for more modern look
    panel_color = '#313244'  # Lighter panel color
    accent_color = '#F5C2E7'  # Vibrant accent
    button_color_primary = '#89B4FA'  # Button primary color
    button_color_secondary = '#74C7EC'  # Button secondary color
    button_color_danger = '#F38BA8'  # Button danger color
    text_color = '#CDD6F4'  # Brighter text
    secondary_text = '#A6ADC8'  # Light gray text

    # Add app icon
    try:
        root.iconbitmap('app_icon.ico')  # If you have an icon file
    except:
        pass  # Skip if no icon file available

    # Create UI elements
    video_funcs, status_bar = create_ui(
        root=root, 
        title_font=title_font, 
        heading_font=heading_font, 
        btn_font=btn_font, 
        label_font=label_font, 
        panel_color=panel_color, 
        bg_color=bg_color, 
        button_color_primary=button_color_primary,
        button_color_secondary=button_color_secondary, 
        button_color_danger=button_color_danger, 
        text_color=text_color, 
        secondary_text=secondary_text, 
        accent_color=accent_color
    )

    # Make sure to stop all threads when the application closes
    def on_closing():
        print("Closing application and stopping all threads...")
        
        video_funcs.stop_video_stream()
        
        # Give some time for threads to clean up
        time.sleep(0.5)
        
        # Destroy the root window
        root.destroy()

    # Handle system signals for clean exit
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, closing application...")
        on_closing()
        sys.exit(0)

    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Run the application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, closing application...")
        on_closing()
    except Exception as e:
        print(f"Error: {str(e)}")
        on_closing()
    finally:
        print("Exiting application")

if __name__ == "__main__":
    main()