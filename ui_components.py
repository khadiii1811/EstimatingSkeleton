import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import threading
from video_processor import run_video_processing, process_combination, process_face_detection, process_hand_tracking, process_face_mesh, process_pose_estimation, process_fall_detection, update_preview_image
from evaluation import show_evaluation_report, show_performance_comparison, reset_evaluation_data, evaluation_data, session_history
from utils import select_video_file
import os
import time
import psutil
import datetime
import statistics

# UI styling functions and helpers
def create_gradient(canvas, width, height, color1, color2, horizontal=True):
    """Create a gradient effect on a canvas"""
    # Convert hex colors to RGB
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    if horizontal:
        for i in range(width):
            # Calculate the gradient color at this position
            r = int(r1 + (r2-r1) * i / width)
            g = int(g1 + (g2-g1) * i / width)
            b = int(b1 + (b2-b1) * i / width)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(i, 0, i, height, fill=color)
    else:
        for i in range(height):
            # Calculate the gradient color at this position
            r = int(r1 + (r2-r1) * i / height)
            g = int(g1 + (g2-g1) * i / height)
            b = int(b1 + (b2-b1) * i / height)
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(0, i, width, i, fill=color)

def draw_skeleton_icon(canvas, x, y, scale=1.0, color="#00CCFF"):
    """Draw a skeleton icon on the canvas"""
    # Head
    canvas.create_oval(x-10*scale, y-25*scale, x+10*scale, y-5*scale, outline=color, width=2)
    
    # Body
    canvas.create_line(x, y-5*scale, x, y+15*scale, fill=color, width=2)
    
    # Arms
    canvas.create_line(x, y, x-15*scale, y+10*scale, fill=color, width=2)
    canvas.create_line(x, y, x+15*scale, y+10*scale, fill=color, width=2)
    
    # Legs
    canvas.create_line(x, y+15*scale, x-10*scale, y+30*scale, fill=color, width=2)
    canvas.create_line(x, y+15*scale, x+10*scale, y+30*scale, fill=color, width=2)
    
    # Add joints as small circles with animation tags
    joint_positions = [
        (x, y-5*scale),  # Neck
        (x, y),          # Shoulders
        (x-15*scale, y+10*scale),  # Left hand
        (x+15*scale, y+10*scale),  # Right hand
        (x, y+15*scale),  # Hip
        (x-10*scale, y+30*scale),  # Left foot
        (x+10*scale, y+30*scale)   # Right foot
    ]
    
    for px, py in joint_positions:
        canvas.create_oval(px-2*scale, py-2*scale, px+2*scale, py+2*scale, 
                           fill=color, outline="", tags="joint")
    
    # Add some detection markers/lines to simulate AI vision
    canvas.create_rectangle(x-20*scale, y-30*scale, x+20*scale, y+35*scale, 
                           outline="#44FFFF", width=1, dash=(5, 3), tags="detection_marker")
    
    # Add some keypoint detection lines with animation tags
    canvas.create_line(x-25*scale, y-15*scale, x-10*scale, y-15*scale, 
                       fill="#44FFFF", width=1, tags="detection_line")
    canvas.create_line(x+10*scale, y-15*scale, x+25*scale, y-15*scale, 
                       fill="#44FFFF", width=1, tags="detection_line")
    canvas.create_line(x-25*scale, y+20*scale, x-15*scale, y+20*scale, 
                       fill="#44FFFF", width=1, tags="detection_line")
    canvas.create_line(x+15*scale, y+20*scale, x+25*scale, y+20*scale, 
                       fill="#44FFFF", width=1, tags="detection_line")

def update_glow(banner_canvas, glow_value):
    """Update glow effect on banner elements"""
    # Create glow effect around detection markers
    glow_color = f"#{30+int(glow_value*2):02x}{220+int(glow_value*3):02x}{255:02x}"
    
    # Update skeleton detection markers with new color
    banner_canvas.itemconfig("detection_marker", outline=glow_color)
    banner_canvas.itemconfig("detection_line", fill=glow_color)
    banner_canvas.itemconfig("joint", fill=glow_color)

def create_button(parent, text, command, color, fg_color, btn_font, width=30):
    """Create a button with hover effect"""
    button_style = {
        'font': btn_font, 
        'borderwidth': 0,
        'relief': RAISED,
        'padx': 15,
        'pady': 10,
        'cursor': 'hand2',  # Hand cursor on hover
        'borderwidth': 0,   # No border for a cleaner look
        'highlightthickness': 0,  # No highlight
        'highlightbackground': '#585B70',  # Border color
        'activebackground': '#F5C2E7',  # Color when clicked
        'activeforeground': '#11111B'  # Text color when clicked
    }
    
    # Create a frame to hold the button for additional styling
    btn_frame = Frame(parent, bg=parent['bg'], padx=2, pady=2)
    
    btn = Button(btn_frame, text=text, command=command,
               bg=color, fg=fg_color, width=width, **button_style)
    btn.pack(fill=X, expand=True)
    
    # Hover effects
    def on_enter(e):
        btn['background'] = '#F5C2E7'  # Common hover color
        btn['foreground'] = '#11111B'
        # Add a subtle zoom effect
        btn.config(padx=16, pady=11)
    
    def on_leave(e):
        btn['background'] = color
        btn['foreground'] = fg_color
        # Return to normal size
        btn.config(padx=15, pady=10)
    
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    
    return btn_frame

# Video function handlers - these connect UI buttons to video processing
class VideoFunctions:
    def __init__(self, root, my_label1, img_container, status_bar):
        self.root = root
        self.my_label1 = my_label1
        self.img_container = img_container
        self.status_bar = status_bar
        self.video_stream = None
        self.video_running = False
        self.current_video_function = None
    
    def stop_video_stream(self):
        """Stop any running video stream"""
        self.video_running = False
        if self.current_video_function is not None:
            self.current_video_function.set()  # Signal thread to stop
        
        if self.video_stream is not None and self.video_stream.is_alive():
            # Wait for thread to finish
            self.video_stream.join(timeout=1.0)
        
        self.current_video_function = None
        # Reset label
        self.my_label1.config(text='No image selected', image="")
        self.status_bar.config(text="Ready")
    
    def run_combination(self):
        """Run combination of all detection methods"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_combination, stop_event, "combination", 
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running combination estimation...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_face_detection(self):
        """Run face detection"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_face_detection, stop_event, "face_detection",
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running face detection...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_hand_tracking(self):
        """Run hand tracking"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_hand_tracking, stop_event, "hand_tracking",
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running hand tracking...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_face_mesh(self):
        """Run face mesh detection"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_face_mesh, stop_event, "face_mesh",
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running facemesh detection...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_pose_estimation(self):
        """Run pose estimation"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_pose_estimation, stop_event, "pose_estimation",
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running real-time pose estimation...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_fall_detection(self):
        """Run fall detection"""
        # Create choice window
        choice_window = Toplevel(self.root)
        choice_window.title("Select video source")
        choice_window.geometry("300x150")
        choice_window.configure(bg='#313244')
        choice_window.resizable(False, False)
        
        # Center the window
        choice_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, 
                                            self.root.winfo_rooty() + 50))
        
        # Add label
        Label(choice_window, text="Select video source for fall detection:", 
              font=('Helvetica', 12), bg='#313244', fg='#CDD6F4').pack(pady=10)
        
        # Add buttons
        btn_font = ('Helvetica', 12)
        
        Button(choice_window, text="From Webcam", font=btn_font, bg='#89B4FA', fg='#CDD6F4',
              command=lambda: [choice_window.destroy(), self.run_fall_detection_from_webcam()]).pack(fill=X, padx=20, pady=5)
        
        Button(choice_window, text="From Video File", font=btn_font, bg='#74C7EC', fg='#CDD6F4',
              command=lambda: [choice_window.destroy(), self.run_fall_detection_from_file()]).pack(fill=X, padx=20, pady=5)
        
        Button(choice_window, text="Cancel", font=btn_font, bg='#F38BA8', fg='#CDD6F4',
              command=choice_window.destroy).pack(fill=X, padx=20, pady=5)
        
        # Focus on window
        choice_window.transient(self.root)
        choice_window.grab_set()
    
    def run_fall_detection_from_webcam(self):
        """Run fall detection from webcam"""
        self.stop_video_stream()
        
        # Set up new video stream
        self.video_running = True
        stop_event = threading.Event()
        
        self.video_stream = threading.Thread(
            target=run_video_processing,
            args=(process_fall_detection, stop_event, "fall_detection",
                  self.img_container, self.my_label1, self.status_bar)
        )
        
        self.current_video_function = stop_event
        self.status_bar.config(text="Running fall detection from webcam...")
        
        # Start video stream
        self.video_stream.daemon = True
        self.video_stream.start()
    
    def run_fall_detection_from_file(self):
        """Run fall detection from video file"""
        from tkinter.filedialog import askopenfilename
        from utils import select_video_file
        from evaluation import reset_evaluation_data, evaluation_data, session_history
        import datetime
        import statistics
        
        # Stop any existing video stream
        self.stop_video_stream()
        
        # Select video file
        video_path = select_video_file("Select video file for fall detection")
        if not video_path:
            return
        
        try:
            # Reset evaluation data for new session
            reset_evaluation_data()
            evaluation_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            evaluation_data["features_used"].append("fall_detection_file")
            
            # Update status
            self.status_bar.config(text=f"Processing video file: {os.path.basename(video_path)}...")
            
            # Create a thread to process the video file
            self.video_running = True
            stop_event = threading.Event()
            
            # Metrics collection variables
            fps_values = []
            frames_processed = 0
            successful_detections = 0
            response_times = []
            cpu_usages = []
            memory_usages = []
            falls_detected = 0
            
            # Create a custom function that will use the selected file
            def process_video_file(frame, pTime):
                return process_fall_detection(frame, pTime)
            
            # Set up custom processing function
            def process_with_file():
                import cv2
                import time
                import psutil
                
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                pTime = 0
                process = psutil.Process(os.getpid())
                
                while not stop_event.is_set() and cap.isOpened():
                    # Measure response time start
                    response_start = time.time()
                    
                    # Monitor resource usage before processing
                    cpu_before = process.cpu_percent()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    success, frame = cap.read()
                    if not success:
                        # End of video
                        self.status_bar.config(text="End of video file")
                        break
                    
                    # Count frames
                    nonlocal frames_processed
                    frames_processed += 1
                    
                    # Process the frame
                    processed_frame = process_fall_detection(frame, pTime)
                    
                    # Check if fall was detected (look for red "FALL DETECTED!" text)
                    # This is a simple way to check - in a real app you'd have a direct API
                    if "FALL DETECTED!" in processed_frame.__str__():
                        nonlocal falls_detected
                        falls_detected += 1
                    
                    # Count successful detection if we got a valid frame back
                    if processed_frame is not None:
                        nonlocal successful_detections
                        successful_detections += 1
                    
                    # Monitor resource usage after processing
                    cpu_after = process.cpu_percent()
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Calculate resource usage for this frame
                    cpu_usage = cpu_after - cpu_before if cpu_after > cpu_before else 0
                    memory_usage = memory_after - memory_before if memory_after > memory_before else 0
                    
                    # Store resource usage metrics
                    nonlocal cpu_usages, memory_usages
                    cpu_usages.append(cpu_usage)
                    memory_usages.append(memory_usage)
                    
                    # Measure response time end
                    response_end = time.time()
                    response_time = (response_end - response_start) * 1000  # Convert to ms
                    nonlocal response_times
                    response_times.append(response_time)
                    
                    # Calculate FPS
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    
                    # Store FPS value
                    nonlocal fps_values
                    fps_values.append(fps)
                    
                    # Add FPS to the frame
                    cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
                    
                    # Update the preview image
                    update_preview_image(processed_frame, self.img_container, self.my_label1)
                    
                    # Introduce a small delay to make it more viewable
                    cv2.waitKey(30)
                    
                    if stop_event.is_set():
                        break
                
                # Release the video
                cap.release()
                
                # Calculate and store metrics
                if frames_processed > 0:
                    if fps_values:
                        evaluation_data["performance"]["fps_values"] = fps_values
                        evaluation_data["performance"]["avg_fps"] = statistics.mean(fps_values)
                        evaluation_data["performance"]["min_fps"] = min(fps_values)
                        evaluation_data["performance"]["max_fps"] = max(fps_values)
                        try:
                            evaluation_data["performance"]["std_fps"] = statistics.stdev(fps_values)
                            
                            # Calculate stability score (0-10)
                            if evaluation_data["performance"]["avg_fps"] > 0:
                                stability_ratio = evaluation_data["performance"]["std_fps"] / evaluation_data["performance"]["avg_fps"]
                                stability_score = max(0, 10 - (stability_ratio * 20))  # Convert to 0-10 scale
                                evaluation_data["performance"]["stability_score"] = stability_score
                            else:
                                evaluation_data["performance"]["stability_score"] = 0
                        except:
                            evaluation_data["performance"]["std_fps"] = 0
                            evaluation_data["performance"]["stability_score"] = 10
                    
                    # Calculate detection rate
                    evaluation_data["accuracy"]["fall_detection_rate"] = falls_detected / frames_processed if frames_processed > 0 else 0
                    evaluation_data["accuracy"]["skeleton_detection_rate"] = successful_detections / frames_processed if frames_processed > 0 else 0
                    
                    # Confidence rating based on detection rate
                    evaluation_data["accuracy"]["detection_confidence"] = evaluation_data["accuracy"]["skeleton_detection_rate"] * 10
                    
                    # Resource usage metrics
                    if cpu_usages:
                        evaluation_data["resource_usage"]["cpu_usage"] = cpu_usages
                        evaluation_data["resource_usage"]["avg_cpu"] = statistics.mean(cpu_usages)
                    
                    if memory_usages:
                        evaluation_data["resource_usage"]["memory_usage"] = memory_usages
                        evaluation_data["resource_usage"]["avg_memory"] = statistics.mean(memory_usages)
                    
                    # Calculate efficiency score (0-10)
                    if evaluation_data["performance"]["avg_fps"] > 0:
                        cpu_factor = max(0.1, evaluation_data["resource_usage"]["avg_cpu"] / 10)
                        mem_factor = max(0.1, evaluation_data["resource_usage"]["avg_memory"] / 200)
                        efficiency = evaluation_data["performance"]["avg_fps"] / (cpu_factor * mem_factor)
                        evaluation_data["resource_usage"]["efficiency_score"] = min(10, efficiency / 10)
                    else:
                        evaluation_data["resource_usage"]["efficiency_score"] = 5
                    
                    # Response time metrics
                    if response_times:
                        evaluation_data["usability"]["response_time"] = statistics.mean(response_times)
                        
                        # Calculate usability score
                        if evaluation_data["usability"]["response_time"] < 30:
                            evaluation_data["usability"]["usability_score"] = 10
                        elif evaluation_data["usability"]["response_time"] < 100:
                            evaluation_data["usability"]["usability_score"] = 8
                        elif evaluation_data["usability"]["response_time"] < 200:
                            evaluation_data["usability"]["usability_score"] = 6
                        elif evaluation_data["usability"]["response_time"] < 300:
                            evaluation_data["usability"]["usability_score"] = 4
                        else:
                            evaluation_data["usability"]["usability_score"] = 2
                    
                    # Calculate overall score
                    performance_score = (evaluation_data["performance"]["avg_fps"] / 30) * 10  # Normalize
                    performance_score = min(10, performance_score)
                    
                    overall_score = (
                        0.4 * ((performance_score + evaluation_data["performance"]["stability_score"]) / 2) +
                        0.3 * evaluation_data["accuracy"]["detection_confidence"] +
                        0.2 * evaluation_data["resource_usage"]["efficiency_score"] +
                        0.1 * evaluation_data["usability"]["usability_score"]
                    )
                    
                    evaluation_data["overall_score"] = overall_score
                    
                    # Save to session history
                    session_history.append(evaluation_data.copy())
                    
                    # Update status
                    self.status_bar.config(text=f"Video processing completed. Score: {overall_score:.2f}/10")
                else:
                    self.status_bar.config(text="No frames were processed")
            
            # Start the thread
            self.video_stream = threading.Thread(target=process_with_file)
            self.current_video_function = stop_event
            
            # Start video stream
            self.video_stream.daemon = True
            self.video_stream.start()
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
            self.my_label1.config(text=f"Error processing video: {str(e)}", image="")
    
    def show_evaluation(self):
        """Show evaluation report"""
        show_evaluation_report(
            self.root, 
            panel_color='#313244', 
            text_color='#CDD6F4', 
            accent_color='#F5C2E7',
            button_color_primary='#89B4FA',
            button_color_danger='#F38BA8',
            btn_font=('Helvetica', 12)
        )
    
    def show_comparison(self):
        """Show performance comparison"""
        show_performance_comparison(
            self.root, 
            panel_color='#313244', 
            text_color='#CDD6F4', 
            accent_color='#F5C2E7',
            button_color_primary='#89B4FA',
            button_color_danger='#F38BA8',
            btn_font=('Helvetica', 12)
        )
    
    def estimate_from_image(self):
        """Estimate skeleton from image"""
        from tkinter.filedialog import askopenfilename
        
        # Stop any existing video stream
        self.stop_video_stream()
        
        link = askopenfilename()
        if not link:
            return
            
        try:
            # Reset evaluation data for new session
            reset_evaluation_data()
            evaluation_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            evaluation_data["features_used"].append("image_pose_estimation")
            
            # Update status
            self.status_bar.config(text="Processing image...")
            
            # Measure performance
            start_time = time.time()
            
            # Monitor resource usage before processing
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process the image for skeleton detection
            cv_img = cv2.imread(link)
            if cv_img is None:
                self.status_bar.config(text="Error: Could not load image")
                return
                
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            from detectors import PoseDetector
            detector = PoseDetector()
            processed_img = detector.findPose(img_rgb)
            lmList, bboxInfo = detector.findPosition(processed_img, bboxWithHands=True)
            
            # Record detection success
            detection_successful = len(lmList) > 0
            
            # End timing and calculate metrics
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # ms
            
            # Monitor resource usage after processing
            cpu_after = process.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate resource usage
            cpu_usage = cpu_after - cpu_before if cpu_after > cpu_before else 0
            memory_usage = memory_after - memory_before if memory_after > memory_before else 0
            
            # Update evaluation data
            fps = 1000 / processing_time if processing_time > 0 else 0
            evaluation_data["performance"]["fps_values"].append(fps)
            evaluation_data["performance"]["avg_fps"] = fps
            evaluation_data["performance"]["min_fps"] = fps
            evaluation_data["performance"]["max_fps"] = fps
            evaluation_data["performance"]["std_fps"] = 0
            evaluation_data["performance"]["stability_score"] = 10  # Single image so perfect stability
            
            if detection_successful:
                evaluation_data["accuracy"]["skeleton_detection_rate"] = 1.0
            else:
                evaluation_data["accuracy"]["skeleton_detection_rate"] = 0.0
                
            evaluation_data["accuracy"]["detection_confidence"] = 10 if detection_successful else 0
            
            evaluation_data["resource_usage"]["cpu_usage"].append(cpu_usage)
            evaluation_data["resource_usage"]["memory_usage"].append(memory_usage)
            evaluation_data["resource_usage"]["avg_cpu"] = cpu_usage
            evaluation_data["resource_usage"]["avg_memory"] = memory_usage
            
            # Calculate efficiency score (1-10)
            if fps > 0:
                cpu_factor = max(0.1, cpu_usage / 10)  # Normalize CPU, avoid div by 0
                mem_factor = max(0.1, memory_usage / 200)  # Normalize Memory, avoid div by 0
                efficiency = fps / (cpu_factor * mem_factor)
                evaluation_data["resource_usage"]["efficiency_score"] = min(10, efficiency / 10)
            else:
                evaluation_data["resource_usage"]["efficiency_score"] = 5  # Default
                
            evaluation_data["usability"]["response_time"] = processing_time
            
            # Calculate usability score based on response time
            if processing_time < 100:  # Under 100ms is ideal for image
                evaluation_data["usability"]["usability_score"] = 10
            elif processing_time < 300:  # Under 300ms is very good
                evaluation_data["usability"]["usability_score"] = 8
            elif processing_time < 500:  # Under 500ms is good
                evaluation_data["usability"]["usability_score"] = 6
            elif processing_time < 1000:  # Under 1s is acceptable
                evaluation_data["usability"]["usability_score"] = 4
            else:  # Over 1s is poor
                evaluation_data["usability"]["usability_score"] = 2
                
            # Calculate overall score
            performance_score = min(10, (fps / 10) * 10)  # Normalize - 10fps is good for image
            
            overall_score = (
                0.4 * ((performance_score + evaluation_data["performance"]["stability_score"]) / 2) +
                0.3 * evaluation_data["accuracy"]["detection_confidence"] +
                0.2 * evaluation_data["resource_usage"]["efficiency_score"] +
                0.1 * evaluation_data["usability"]["usability_score"]
            )
            
            evaluation_data["overall_score"] = overall_score
            
            # Save to session history
            session_history.append(evaluation_data.copy())
            
            # Convert back to BGR for display
            processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            
            # Update the preview
            update_preview_image(processed_img_bgr, self.img_container, self.my_label1)
            
            # Update status
            self.status_bar.config(text=f"Skeleton estimated successfully (Processing time: {processing_time:.2f}ms, Score: {overall_score:.2f}/10)")
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")

def create_ui(root, title_font, heading_font, btn_font, label_font, panel_color, bg_color, button_color_primary, 
             button_color_secondary, button_color_danger, text_color, secondary_text, accent_color):
    """Create the complete UI"""
    
    # Create header frame with app title and group info
    header_frame = Frame(root, bg="#1A1B26", pady=0, padx=0)
    header_frame.pack(fill=X)
    
    # Create a canvas for the banner
    banner_canvas = Canvas(header_frame, bg="#1A1B26", height=100, highlightthickness=0)
    banner_canvas.pack(fill=X, expand=True)
    
    # Function to update banner size on window resize
    def update_banner_size(event=None):
        width = root.winfo_width()
        if width <= 1:  # Not yet drawn
            width = 1280
        
        # Clear the canvas and redraw with new width
        banner_canvas.delete("all")
        create_gradient(banner_canvas, width, 100, "#1A1A2E", "#16213E", horizontal=False)
        
        # Re-add pattern overlay with more density for a modern look
        for i in range(0, width, 15):
            for j in range(0, 100, 15):
                if (i + j) % 30 == 0:
                    dot_size = 1.5 if (i + j) % 60 == 0 else 1
                    banner_canvas.create_oval(i, j, i+dot_size, j+dot_size, 
                                            fill="#FFFFFF", outline="", width=0, stipple="gray12")
        
        # Add a more prominent left accent bar
        banner_canvas.create_rectangle(0, 0, 15, 100, fill="#7C83FD", outline="")
        
        # Add a subtle glow effect behind the title
        banner_canvas.create_oval(width/2-150, 20, width/2+150, 60, 
                                fill="#4C4CFF", outline="", stipple="gray25")
        
        # Draw skeleton icons with slight glow
        draw_skeleton_icon(banner_canvas, 70, 50, scale=1.3)
        draw_skeleton_icon(banner_canvas, width - 70, 50, scale=1.3)
        
        # Add project title with enhanced shadow effect
        banner_canvas.create_text(width/2 + 2, 32, text="Human Skeleton Estimating", 
                                font=('Montserrat', 26, 'bold'), fill="#111122", anchor=CENTER)
        banner_canvas.create_text(width/2, 30, text="Human Skeleton Estimating", 
                                font=('Montserrat', 26, 'bold'), fill="#F5F5F5", anchor=CENTER)
        
        # Add subtitle - Final Project with slight glow
        banner_canvas.create_text(width/2 + 1, 56, text="Final Project", 
                                font=('Montserrat', 12, 'italic'), fill="#222236", anchor=CENTER)
        banner_canvas.create_text(width/2, 55, text="Final Project", 
                                font=('Montserrat', 12, 'italic'), fill="#A2A8D3", anchor=CENTER)
        
        # Add team information with better contrast
        banner_canvas.create_rectangle(width/2 - 500, 70, width/2 + 500, 77, 
                                     fill="#111122", outline="", stipple="gray50")
        banner_canvas.create_text(width/2, 75, 
                             text="Team: Nguyễn Nguyên Toàn Khoa (22110044) | Nguyen Hoàng Huy (22110036) | Nguyễn Lê Tùng Chi (22110013) | Nguyên Thành Tính (22110077)", 
                             font=('Montserrat', 12), fill="#E0E4FF", anchor=CENTER)
        
        # Add horizontal divider line at the bottom with a glowing effect
        banner_canvas.create_line(0, 98, width, 98, fill="#5A6B9F", width=2, dash=(2, 2))
        banner_canvas.create_line(0, 99, width, 99, fill="#394867", width=1)
        
        # Add a decorative element to the right
        banner_canvas.create_rectangle(width-15, 0, width, 100, fill="#7C83FD", outline="")
    
    # Bind to window resize
    root.bind("<Configure>", update_banner_size)
    
    # Initial drawing of banner
    update_banner_size()
    
    # Define animation variables
    glow_value = 0 
    glow_direction = 1  # 1 = increasing, -1 = decreasing
    glow_speed = 0.5
    
    # Define a function for animated elements
    def animate_banner():
        nonlocal glow_direction, glow_value, glow_speed
        
        # Update glow effect
        if glow_direction > 0:
            glow_value += glow_speed
            if glow_value >= 10:
                glow_direction = -1
        else:
            glow_value -= glow_speed
            if glow_value <= 0:
                glow_direction = 1
        
        # Apply glow effect to skeleton joints
        update_glow(banner_canvas, glow_value)
        
        # Schedule the next animation frame
        root.after(100, animate_banner)
    
    # Start the animation after a short delay to ensure UI is ready
    root.after(500, animate_banner)
    
    # Create a frame for the main content area with a 3D effect and rounded corners
    content_frame = Frame(root, bg=panel_color, bd=1, relief=FLAT,
                        padx=25, pady=25)
    content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)
    
    # Add subtle shadow effect to the content frame
    content_shadow = Canvas(content_frame, bg=panel_color, height=5, highlightthickness=0)
    content_shadow.pack(fill=X, side=BOTTOM)
    create_gradient(content_shadow, 1280, 5, panel_color, "#11111B", horizontal=False)
    
    # Add a left panel for the image display
    frame1 = LabelFrame(content_frame, text='Image Preview', font=heading_font, 
                      bg=panel_color, fg=accent_color, bd=1, labelanchor='n',
                      padx=15, pady=15, relief=FLAT)
    frame1.pack(fill=BOTH, side=LEFT, expand=True, padx=10, pady=10)
    
    # Image container with improved visual styling
    img_container = Frame(frame1, bg='#1E1E2E', width=500, height=500)
    img_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
    img_container.pack_propagate(False)  # Prevent container from shrinking
    
    # Add subtle border to image container
    img_border = Canvas(img_container, bg='#1E1E2E', bd=0, highlightthickness=1, 
                       highlightbackground=accent_color)
    img_border.pack(fill=BOTH, expand=True, padx=1, pady=1)
    
    my_label1 = Label(img_border, text='No image selected', font=label_font,
                    bg='#181825', fg=secondary_text, width=40, height=20)
    my_label1.pack(fill=BOTH, expand=True)
    
    # Right side panel with tabs for different functionality
    right_panel = Frame(content_frame, bg=panel_color)
    right_panel.pack(fill=BOTH, side=RIGHT, expand=True, padx=10, pady=10)
    
    # Create two panels for the functions - using themed tabs
    tab_header = Frame(right_panel, bg=panel_color)
    tab_header.pack(fill=X)
    
    tab_active_bg = accent_color
    tab_inactive_bg = '#45475A'
    
    # Create frames for tab content with improved padding
    frame2 = Frame(right_panel, bg=panel_color, padx=20, pady=30)
    frame3 = Frame(right_panel, bg=panel_color, padx=20, pady=30)
    
    # Function to switch between tabs with animation
    def activate_tab(tab_num):
        if tab_num == 1:
            tab1_btn.config(bg=tab_active_bg, fg='#11111B')
            tab2_btn.config(bg=tab_inactive_bg, fg=secondary_text)
            # Add a smooth transition effect
            frame3.pack_forget()
            frame2.pack(fill=BOTH, expand=True)
        else:
            tab1_btn.config(bg=tab_inactive_bg, fg=secondary_text)
            tab2_btn.config(bg=tab_active_bg, fg='#11111B')
            # Add a smooth transition effect
            frame2.pack_forget()
            frame3.pack(fill=BOTH, expand=True)
    
    # Create tab buttons with better styling
    tab1_btn = Button(tab_header, text='SKELETON ESTIMATION', font=btn_font,
                    bg=tab_active_bg, fg='#11111B', bd=0, padx=15, pady=12,
                    relief=FLAT, activebackground='#F5C2E7', cursor='hand2',
                    command=lambda: activate_tab(1))
    tab1_btn.pack(side=LEFT, fill=X, expand=True)
    
    tab2_btn = Button(tab_header, text='EXTENDED FUNCTIONS', font=btn_font,
                    bg=tab_inactive_bg, fg=secondary_text, bd=0, padx=15, pady=12,
                    relief=FLAT, activebackground='#F5C2E7', cursor='hand2',
                    command=lambda: activate_tab(2))
    tab2_btn.pack(side=LEFT, fill=X, expand=True)
    
    # Status bar with improved styling
    status_frame = Frame(root, bg='#11111B', padx=1, pady=1)
    status_frame.pack(side=BOTTOM, fill=X)
    
    status_bar = Label(status_frame, text="Ready", bd=0, relief=FLAT, anchor=W, 
                      bg='#181825', fg=secondary_text, font=('Helvetica', 10), pady=6, padx=10)
    status_bar.pack(side=BOTTOM, fill=X)
    
    # Create video functions handler
    video_funcs = VideoFunctions(root, my_label1, img_container, status_bar)
    
    # Primary buttons - main functions for tab 1 with improved spacing
    button_image = create_button(frame2, '📷 Estimate Skeleton from Image', 
                              video_funcs.estimate_from_image, button_color_primary, text_color, btn_font)
    button_image.pack(pady=12, fill=X)
    
    button_estm_rt = create_button(frame2, '📹 Estimate Skeleton (Real-time)', 
                                video_funcs.run_pose_estimation, button_color_primary, text_color, btn_font)
    button_estm_rt.pack(pady=12, fill=X)
    
    # Add a button to stop the video stream with bigger icon
    button_stop = create_button(frame2, '⏹️ Stop Current Stream', 
                              video_funcs.stop_video_stream, button_color_danger, text_color, btn_font)
    button_stop.pack(pady=12, fill=X)
    
    # Add a separator with improved style
    separator = Frame(frame2, height=2, bg='#585B70')
    separator.pack(fill=X, pady=25)
    
    # Add subtle shine lines to separator
    sep_canvas = Canvas(frame2, height=4, bg=panel_color, highlightthickness=0)
    sep_canvas.pack(fill=X, pady=0)
    sep_canvas.create_line(0, 0, 800, 0, fill='#585B70', width=1)
    sep_canvas.create_line(0, 3, 800, 3, fill='#313244', width=1)
    
    button_quit = create_button(frame2, '❌ Exit Program', 
                              root.destroy, button_color_danger, text_color, btn_font)
    button_quit.pack(pady=12, fill=X)
    
    # Extended function buttons for tab 2 with improved spacing and icons
    button_face = create_button(frame3, '👤 Face Detection', 
                              video_funcs.run_face_detection, button_color_secondary, text_color, btn_font)
    button_face.pack(pady=12, fill=X)
    
    button_hand = create_button(frame3, '✋ Hand Tracking', 
                              video_funcs.run_hand_tracking, button_color_secondary, text_color, btn_font)
    button_hand.pack(pady=12, fill=X)
    
    button_facemesh = create_button(frame3, '😀 Face Mesh Detection', 
                                 video_funcs.run_face_mesh, button_color_secondary, text_color, btn_font)
    button_facemesh.pack(pady=12, fill=X)
    
    button_combo = create_button(frame3, '🔄 Combination Estimate', 
                               video_funcs.run_combination, button_color_secondary, text_color, btn_font)
    button_combo.pack(pady=12, fill=X)
    
    # Make fall detection button stand out more
    button_fall = create_button(frame3, '🚨 Fall Detection (Ngã)', 
                              video_funcs.run_fall_detection, accent_color, '#11111B', btn_font)
    button_fall.pack(pady=12, fill=X)
    
    # Add separator with improved style
    separator2 = Frame(frame3, height=2, bg='#585B70')
    separator2.pack(fill=X, pady=25)
    
    # Add subtle shine lines to separator
    sep_canvas2 = Canvas(frame3, height=4, bg=panel_color, highlightthickness=0)
    sep_canvas2.pack(fill=X, pady=0)
    sep_canvas2.create_line(0, 0, 800, 0, fill='#585B70', width=1)
    sep_canvas2.create_line(0, 3, 800, 3, fill='#313244', width=1)
    
    # Add evaluation button with more prominent design
    button_evaluation = create_button(frame3, '📊 Performance Analysis & Evaluation', 
                                    video_funcs.show_evaluation, button_color_primary, text_color, btn_font)
    button_evaluation.pack(pady=12, fill=X)
    
    # Add comparison button with more prominent design
    button_comparison = create_button(frame3, '📈 Performance Comparison Chart', 
                                    video_funcs.show_comparison, '#85C88A', text_color, btn_font)
    button_comparison.pack(pady=12, fill=X)
    
    # Initially show the first tab
    activate_tab(1)
    
    return video_funcs, status_bar 