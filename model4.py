import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time
from openpyxl import Workbook
import sys
import os
import subprocess
import pandas as pd  # Import pandas for interpolation

# Load the pre-trained YOLOv8 model
model = YOLO('yolo11n.pt')

# Define classes of interest
VEHICLE_CLASSES = {'motorbike', 'truck', 'bus', 'car' ,'motorcycle'}

def get_class_ids(model, class_names):
    """Retrieve class IDs for the given class names."""
    class_ids = set()
    for key, value in model.names.items():
        if value in class_names:
            class_ids.add(key)
    return class_ids

vehicle_class_ids = get_class_ids(model, VEHICLE_CLASSES)

class KalmanFilter:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0, 0], dtype='float64')  # Initial position and velocity
        self.F = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]])
        self.Q = np.diag([2.0, 2.0, 0.5, 0.5])  # Process noise
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]])
        self.R = np.diag([0.1, 0.1])  # Measurement noise
        self.P = np.eye(4) * 200  # Initial uncertainty

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, measurement):
        z = np.array(measurement)
        y = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state += np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_velocity(self):
        return np.linalg.norm(self.state[2:])  # Velocity magnitude

class KalmanVehicleTracker:
    def __init__(self, memory_limit=50):  # Increased memory limit for occlusion handling
        self.tracks = {}
        self.memory_limit = memory_limit
        self.next_track_id = 1
        self.lost_tracks = {}
        self.velocity_history = {}

    def iou(self, box1, box2):
        """Compute IoU between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1, yi1 = max(x1, x3), max(y1, y3)
        xi2, yi2 = min(x2, x4), min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def predict_and_update(self, detections, boxes):
        updated_tracks = {}
        unmatched_detections = list(zip(detections, boxes))

        # Try to match existing tracks
        for track_id, kf in list(self.tracks.items()):
            predicted_pos = kf.predict()
            best_match = None
            best_iou = 0.0

            for detection, box in unmatched_detections:
                iou_score = self.iou(box, [predicted_pos[0] - 10, predicted_pos[1] - 10, predicted_pos[0] + 10, predicted_pos[1] + 10])
                distance = np.linalg.norm(np.array(predicted_pos) - np.array(detection))

                if iou_score > best_iou and distance < 50:  # Use both IoU and distance
                    best_iou = iou_score
                    best_match = (detection, box)

            if best_match:
                kf.update(best_match[0])
                updated_tracks[track_id] = kf
                unmatched_detections.remove(best_match)
            else:
                # Move track to lost_tracks if not matched
                self.lost_tracks[track_id] = (kf, self.memory_limit)


  # Check if lost tracks can be re-associated
        for track_id, (kf, frames_left) in list(self.lost_tracks.items()):
            predicted_pos = kf.predict()
            best_match = None
            best_iou = 0.0

            for detection, box in unmatched_detections:
                iou_score = self.iou(box, [predicted_pos[0] - 10, predicted_pos[1] - 10, predicted_pos[0] + 10, predicted_pos[1] + 10])
                distance = np.linalg.norm(np.array(predicted_pos) - np.array(detection))

                if iou_score > best_iou and distance < 50:
                    best_iou = iou_score
                    best_match = (detection, box)

            if best_match:
                kf.update(best_match[0])
                updated_tracks[track_id] = kf
                unmatched_detections.remove(best_match)
                del self.lost_tracks[track_id]  # Remove from lost tracks
            else:
                if frames_left > 0:
                    self.lost_tracks[track_id] = (kf, frames_left - 1)
                else:
                    del self.lost_tracks[track_id]  # Remove if lost for too

        for detection, box in unmatched_detections:
            kf = KalmanFilter(detection[0], detection[1])
            updated_tracks[self.next_track_id] = kf
            self.next_track_id += 1
        self.tracks = updated_tracks
        return self.tracks

tracker = KalmanVehicleTracker(memory_limit=50)

def format_timestamp(frame_count, fps):
    total_seconds = frame_count / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
    

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = video_path.replace('.mp4', '_processed.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    output_file_name = output_path.replace('.mp4', '_data.xlsx')
    wb = Workbook()
    frame_count = 0
    camera_height = 13
    camera_angle = -52
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames in Input:{total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame ,conf= 0.25,imgsz= 1280) 
        detected_objects = []
        detected_boxes = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            if class_id in vehicle_class_ids:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detected_objects.append((center_x, center_y))
                detected_boxes.append((x1, y1, x2, y2))

        tracks = tracker.predict_and_update(detected_objects, detected_boxes)

        timestamp = format_timestamp(frame_count, fps)

        velocity_data = []  # Store the velocity for interpolation

        for track_id, kf in tracks.items():
            pred_x, pred_y = kf.state[:2]
            velocity = kf.get_velocity()
            scaled_velocity = velocity / (np.cos(np.radians(camera_angle)) * camera_height)
            velocity = max(scaled_velocity - 0.5, 0)

            velocity_data.append((timestamp, track_id, velocity))

            sheet_name = f'Track ID {track_id}'
            if sheet_name not in wb.sheetnames:
                ws = wb.create_sheet(sheet_name)
                ws.append(["Time", "Velocity", "Relative Velocity"])
            else:
                ws = wb[sheet_name]

    # Count the number of vehicles in the current frame
        num_vehicles_current_frame = len(tracker.tracks)
        
        # Initialize unique vehicle IDs set
        if not hasattr(process_video, "unique_vehicle_ids"):
            process_video.unique_vehicle_ids = set()
        
        # Add current vehicle IDs to unique set
        for track_id in tracker.tracks.keys():
            process_video.unique_vehicle_ids.add(track_id)

        # Count total unique vehicles detected so far
        num_vehicles_total = len(process_video.unique_vehicle_ids)

        # Display text on the video frame
        text1 = f"Number of vehicles in current frame: {num_vehicles_current_frame}"
        text2 = f"Number of vehicles till now: {num_vehicles_total}"
        
        # Put text at bottom right of the video
        text_x = width - 450
        text_y = height - 50
        
        cv2.putText(frame, text1, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, text2, (text_x, text_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"Frame {frame_count}: Current vehicles={num_vehicles_current_frame}, Total detected={num_vehicles_total}")


        for track_id, kf in tracks.items():
            pred_x, pred_y = kf.state[:2]
            velocity = kf.get_velocity()

            for (center_x, center_y), (x1, y1, x2, y2) in zip(detected_objects, detected_boxes):
                if abs(pred_x - center_x) < 30 and abs(pred_y - center_y) < 30:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f'ID: {track_id} Vel: {velocity:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        # Interpolate missing velocity values
        velocity_df = pd.DataFrame(velocity_data, columns=["Time", "Track ID", "Velocity"])
        velocity_df['Velocity'] = velocity_df['Velocity'].replace(0, np.nan)  # Replace 0 with NaN for interpolation
        velocity_df['Velocity'] = velocity_df['Velocity'].interpolate(method='linear')  # Apply interpolation
        
        # Re-append interpolated velocities back to the Excel sheet
        for _, row in velocity_df.iterrows():
            track_id = int(row["Track ID"])
            timestamp = row["Time"]
            velocity = row["Velocity"]
            sheet_name = f'Track ID {track_id}'
            ws = wb[sheet_name]
            relative_velocity = max(np.mean([np.linalg.norm(other_kf.state[2:]) for other_id, other_kf in tracks.items() if other_id != track_id]), 0)
            ws.append([timestamp, velocity, relative_velocity])
            out.write(frame)

        frame_count += 1
        Percentage_completed = (frame_count / total_frames) * 100
        if frame_count % 10 == 0:  # Reduce verbosity
         print(f"Processing frame {frame_count}/{total_frames} - {Percentage_completed:.2f}% completed", flush=True)


    cap.release()
    out.release()
    wb.save(output_file_name)
    wb.close()
    print("Processing complete.", flush=True)
    full_output_file_path = os.path.abspath(output_file_name)
    subprocess.run([sys.executable, 'modelExcel.py', full_output_file_path])
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_video(sys.argv[1])
    else:
        print("No video path provided.")
    