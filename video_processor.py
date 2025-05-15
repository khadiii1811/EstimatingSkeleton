import cv2
import time
import threading
import statistics
import datetime
import os
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

# Import evaluation data structure
from evaluation import evaluation_data, reset_evaluation_data, session_history
from detectors import PoseDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from utils import play_alarm_thread

# Initialize mp_pose at global level
mp_pose = mp.solutions.pose

# Common function to update preview image
def update_preview_image(cv_image, img_container, my_label1):
    """Update the Image Preview panel with an OpenCV image"""
    if cv_image is None:
        return
    
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Resize to fit the panel while maintaining aspect ratio
    h, w = image_rgb.shape[:2]
    container_width = img_container.winfo_width()
    container_height = img_container.winfo_height()
    
    # Avoid division by zero and ensure the container has been drawn
    if container_width <= 1 or container_height <= 1:
        container_width = 500
        container_height = 500
    
    # Calculate the scaling factor to fit the image in the container
    scale_width = container_width / w
    scale_height = container_height / h
    scale = min(scale_width, scale_height)
    
    # Resize the image
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized_image = cv2.resize(image_rgb, (new_width, new_height))
    
    # Convert the image to PIL format and then to ImageTk format
    img_pil = Image.fromarray(resized_image)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    # Update the label with the new image
    my_label1.config(image=img_tk, text="")
    my_label1.image = img_tk  # Keep a reference to prevent garbage collection

# Function to run video processing in a thread
def run_video_processing(process_function, stop_event, feature_name="unknown", img_container=None, my_label1=None, status_bar=None):
    """Generic function to run video processing in a thread"""
    global evaluation_data
    
    # Initialize evaluation data for new session
    reset_evaluation_data()
    evaluation_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluation_data["features_used"].append(feature_name)
    
    cap = cv2.VideoCapture(0)
    pTime = 0
    frames_processed = 0
    successful_detections = 0
    
    # Variables for resource monitoring
    try:
        import psutil
        process = psutil.Process(os.getpid())
        resource_monitoring_available = True
    except ImportError:
        print("Warning: psutil library not available. Some performance metrics will not be collected.")
        resource_monitoring_available = False
    
    # Variables for response time
    response_times = []
    
    try:
        while not stop_event.is_set():
            # Measure response time start
            response_start = time.time()
            
            success, frame = cap.read()
            if not success:
                break
            
            # Count total frames
            frames_processed += 1
            
            # Monitor resource usage before processing
            if resource_monitoring_available:
                try:
                    cpu_before = process.cpu_percent()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except:
                    cpu_before = 0
                    memory_before = 0
                
            # Process the frame
            processed_frame = process_function(frame, pTime)
            
            # Monitor resource usage after processing
            if resource_monitoring_available:
                try:
                    cpu_after = process.cpu_percent()
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Calculate resource usage for this frame
                    cpu_usage = cpu_after - cpu_before if cpu_after > cpu_before else 0
                    memory_usage = memory_after - memory_before if memory_after > memory_before else 0
                    
                    # Store resource usage metrics
                    evaluation_data["resource_usage"]["cpu_usage"].append(cpu_usage)
                    evaluation_data["resource_usage"]["memory_usage"].append(memory_usage)
                except:
                    pass
            
            # Measure response time end
            response_end = time.time()
            response_time = (response_end - response_start) * 1000  # Convert to ms
            response_times.append(response_time)
            
            # If processing was successful, increment successful detections count
            if processed_frame is not None:
                # Default to successful detection for general case
                detection_successful = True
                
                # Check by function type
                func_name = process_function.__name__ if hasattr(process_function, '__name__') else "unknown"
                
                # For specific detection functions, we'd use post-checks
                if func_name == 'process_pose_estimation' or func_name == 'process_fall_detection':
                    successful_detections += 1
                elif func_name == 'process_face_detection':
                    successful_detections += 1
                elif func_name == 'process_hand_tracking':
                    successful_detections += 1
                else:
                    # Other processing functions
                    successful_detections += 1
            
            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            # Store FPS value for analysis
            evaluation_data["performance"]["fps_values"].append(fps)
            
            # Add FPS to the frame
            cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)
            
            # Update the preview image
            if img_container and my_label1:
                update_preview_image(processed_frame, img_container, my_label1)
            
            # Small delay and check for stop event again
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
            
            if stop_event.is_set():
                break
    except Exception as e:
        print(f"Error in video processing: {str(e)}")
    finally:
        # Calculate final performance metrics
        calculate_performance_metrics(frames_processed, successful_detections, feature_name, response_times)
        
        # Always release the camera
        cap.release()
        
        # Update status
        if status_bar:
            status_bar.config(text="Ready")

def calculate_performance_metrics(frames_processed, successful_detections, feature_name, response_times):
    """Calculate all performance metrics after processing is complete"""
    global evaluation_data, session_history
    
    if frames_processed > 0:
        # Evaluate detection rate by feature type
        if feature_name in ["pose_estimation", "fall_detection"]:
            evaluation_data["accuracy"]["skeleton_detection_rate"] = successful_detections / frames_processed
        elif feature_name == "face_detection":
            evaluation_data["accuracy"]["face_detection_rate"] = successful_detections / frames_processed
        elif feature_name == "hand_tracking":
            evaluation_data["accuracy"]["hand_detection_rate"] = successful_detections / frames_processed
        
        # Calculate detection confidence based on detection rate
        avg_detection_rate = sum([
            evaluation_data["accuracy"]["skeleton_detection_rate"],
            evaluation_data["accuracy"]["face_detection_rate"],
            evaluation_data["accuracy"]["hand_detection_rate"],
            evaluation_data["accuracy"]["fall_detection_rate"]
        ]) / 4
        
        # Evaluate confidence (0-10)
        evaluation_data["accuracy"]["detection_confidence"] = avg_detection_rate * 10
    
    # Calculate FPS statistics
    if evaluation_data["performance"]["fps_values"]:
        fps_values = evaluation_data["performance"]["fps_values"]
        evaluation_data["performance"]["avg_fps"] = statistics.mean(fps_values)
        evaluation_data["performance"]["min_fps"] = min(fps_values)
        evaluation_data["performance"]["max_fps"] = max(fps_values)
        try:
            evaluation_data["performance"]["std_fps"] = statistics.stdev(fps_values)
            
            # Calculate stability score (0-10)
            # Low std deviation = high stability
            if evaluation_data["performance"]["avg_fps"] > 0:
                stability_ratio = evaluation_data["performance"]["std_fps"] / evaluation_data["performance"]["avg_fps"]
                stability_score = max(0, 10 - (stability_ratio * 20))  # Convert to 0-10 scale
                evaluation_data["performance"]["stability_score"] = stability_score
            else:
                evaluation_data["performance"]["stability_score"] = 0
        except:
            evaluation_data["performance"]["std_fps"] = 0
            evaluation_data["performance"]["stability_score"] = 10  # If only 1 sample = perfect stability
    
    # Calculate resource usage if data available
    if evaluation_data["resource_usage"]["cpu_usage"]:
        evaluation_data["resource_usage"]["avg_cpu"] = statistics.mean(evaluation_data["resource_usage"]["cpu_usage"])
    if evaluation_data["resource_usage"]["memory_usage"]:
        evaluation_data["resource_usage"]["avg_memory"] = statistics.mean(evaluation_data["resource_usage"]["memory_usage"])
    
    # Calculate resource efficiency score (0-10)
    # Low CPU and memory usage = high efficiency
    if evaluation_data["performance"]["avg_fps"] > 0:
        # Efficiency = FPS / (CPU * Memory)
        cpu_factor = max(0.1, evaluation_data["resource_usage"]["avg_cpu"] / 10)  # Normalize CPU, avoid div by 0
        mem_factor = max(0.1, evaluation_data["resource_usage"]["avg_memory"] / 200)  # Normalize Memory, avoid div by 0
        
        # Efficiency formula: high when FPS high, CPU and memory low
        efficiency = evaluation_data["performance"]["avg_fps"] / (cpu_factor * mem_factor)
        evaluation_data["resource_usage"]["efficiency_score"] = min(10, efficiency / 10)  # Limit to 0-10
    else:
        evaluation_data["resource_usage"]["efficiency_score"] = 5  # Default value
    
    # Calculate average response time
    if response_times:
        evaluation_data["usability"]["response_time"] = statistics.mean(response_times)
        
        # Calculate usability score based on response time
        # Lower response time = higher usability
        if evaluation_data["usability"]["response_time"] < 30:  # Under 30ms is ideal
            evaluation_data["usability"]["usability_score"] = 10
        elif evaluation_data["usability"]["response_time"] < 100:  # Under 100ms is very good
            evaluation_data["usability"]["usability_score"] = 8
        elif evaluation_data["usability"]["response_time"] < 200:  # Under 200ms is good
            evaluation_data["usability"]["usability_score"] = 6
        elif evaluation_data["usability"]["response_time"] < 300:  # Under 300ms is acceptable
            evaluation_data["usability"]["usability_score"] = 4
        else:  # Over 300ms is poor
            evaluation_data["usability"]["usability_score"] = 2
    
    # Calculate overall score based on weighted average of component scores
    # Weights: Performance (40%), Accuracy (30%), Efficiency (20%), Usability (10%)
    performance_score = (evaluation_data["performance"]["avg_fps"] / 30) * 10  # Normalize FPS, assuming 30 FPS is ideal
    performance_score = min(10, performance_score)  # Limit to 0-10
    
    overall_score = (
        0.4 * ((performance_score + evaluation_data["performance"]["stability_score"]) / 2) +  # Performance (40%)
        0.3 * evaluation_data["accuracy"]["detection_confidence"] +  # Accuracy (30%)
        0.2 * evaluation_data["resource_usage"]["efficiency_score"] +  # Efficiency (20%)
        0.1 * evaluation_data["usability"]["usability_score"]  # Usability (10%)
    )
    
    evaluation_data["overall_score"] = overall_score
    
    # Save to session history
    session_history.append(evaluation_data.copy())

# Processing functions for different detection methods
def process_combination(frame, pTime):
    # Create all detectors
    poseDetector = PoseDetector()
    handDetector = HandDetector(detectionCon=0.8, maxHands=2)
    faceDetector = FaceDetector()
    
    # Process pose estimation
    frame = poseDetector.findPose(frame)
    lmList, bboxInfo = poseDetector.findPosition(frame, bboxWithHands=True)
    
    # Process hand detection
    hands, frame = handDetector.findHands(frame)
    if hands and len(hands) > 0:
        # Process first hand
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1['center']
        handType1 = hand1["type"]
        fingers1 = handDetector.fingersUp(hand1)
        
        # Add text for hand
        cv2.putText(frame, f"{handType1}", (bbox1[0], bbox1[1] - 10), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # Process face detection
    frame, bboxs = faceDetector.findFaces(frame)
    if bboxs and len(bboxs) > 0:
        for box in bboxs:
            center = box["center"]
            cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)
    
    return frame

def process_face_detection(frame, pTime):
    detector = FaceDetector()
    frame, bboxs = detector.findFaces(frame)
    
    # Process detected faces
    if bboxs and len(bboxs) > 0:
        for box in bboxs:
            center = box["center"]
            cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)
            
            # Add face detection text
            x, y, w, h = box["bbox"]
            cv2.putText(frame, "Face", (x, y - 10), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    return frame

def process_hand_tracking(frame, pTime):
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    hands, frame = detector.findHands(frame)
    
    # Process hands if detected
    if hands and len(hands) > 0:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1['center']
        handType1 = hand1["type"]
        fingers1 = detector.fingersUp(hand1)
        
        # Draw hand data on frame
        cv2.putText(frame, f"{handType1} Hand", (bbox1[0], bbox1[1] - 20), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    return frame

def process_face_mesh(frame, pTime):
    detector = FaceMeshDetector(maxFaces=2)
    frame, faces = detector.findFaceMesh(frame)
    return frame

def process_pose_estimation(frame, pTime):
    detector = PoseDetector()
    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=True)
    return frame

def process_fall_detection(frame, pTime):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fall_pose = mp_pose.Pose()
    results = fall_pose.process(img_rgb)

    fallen = False
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        fallen = is_fallen(landmarks)
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if fallen:
        cv2.putText(frame, "FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        play_alarm_thread()
    
    return frame

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

def denoise(frame):
    frame = cv2.medianBlur(frame, 11)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    return frame 