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

# Kalman Filter and Vehicle Tracker classes (unchanged)

class KalmanFilter:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0, 0], dtype='float64')  # Initial position and velocity
        self.F = np.array([[1, 0, 1, 0], 
                           [0, 1, 0, 1], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]])
        # Adjusted process noise covariance for higher adaptability
        self.Q = np.diag([2.0, 2.0, 0.5, 0.5])  # Larger values for position to account for speed
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]])
        # Adjusted measurement noise covariance to allow faster changes
        self.R = np.diag([0.1, 0.1])  # Slightly higher noise tolerance
        self.P = np.eye(4) * 200  # Increased initial uncertainty for high-speed objects

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
        # Return velocity magnitude using the predicted velocities in x and y directions
        return np.linalg.norm(self.state[2:])

class KalmanVehicleTracker:
    def __init__(self, memory_limit=400):  # Extended memory limit for high-speed vehicles
        self.tracks = {}
        self.memory_limit = memory_limit
        self.next_track_id = 1
        self.lost_tracks = {}
        self.velocity_history = {}

    def predict_and_update(self, detections):
        updated_tracks = {}
        unmatched_detections = list(detections)

        for track_id, kf in self.tracks.items():
            predicted_pos = kf.predict()
            best_match = None
            best_distance = float('inf')

            # Adjusted association threshold for high-speed motion
            for detection in unmatched_detections:
                distance = np.linalg.norm(np.array(detection) - predicted_pos)
                if distance < best_distance and distance < 150:  # Increased from 75 to 150
                    best_distance = distance
                    best_match = detection

            if best_match is not None:
                kf.update(best_match)
                updated_tracks[track_id] = kf
                unmatched_detections.remove(best_match)

                velocity = kf.get_velocity()
                if track_id not in self.velocity_history:
                    self.velocity_history[track_id] = []
                self.velocity_history[track_id].append(velocity)

        for detection in unmatched_detections:
            reassociated = False
            for track_id, (kf, age) in self.lost_tracks.items():
                if age < self.memory_limit:
                    predicted_pos = kf.predict()
                    distance = np.linalg.norm(np.array(detection) - predicted_pos)
                    if distance < 150:  # Increased from 75 to 150
                        kf.update(detection)
                        updated_tracks[track_id] = kf
                        reassociated = True
                        break

            if not reassociated:
                kf = KalmanFilter(detection[0], detection[1])
                updated_tracks[self.next_track_id] = kf
                self.next_track_id += 1

        for track_id, kf in self.tracks.items():
            if track_id not in updated_tracks:
                self.lost_tracks[track_id] = (kf, 0)
        self.lost_tracks = {track_id: (kf, age + 1) for track_id, (kf, age) in self.lost_tracks.items() if age < self.memory_limit}

        self.tracks = updated_tracks
        return self.tracks

tracker = KalmanVehicleTracker(memory_limit=400)

def format_timestamp(frame_count, fps):
    total_seconds = frame_count / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_video(video_path):
    print(f"Processing video at: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # Use 'mp4v' codec for better macOS compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    output_file_name = output_path.replace('.mp4', '_data.xlsx')
    wb = Workbook()
    frame_count = 0
    camera_height = 13
    camera_angle = -52
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                                                                              
    motorbike_class_id = None
    person_class_id = None
    cow_class_id= None
    for key, value in model.names.items():
        if value in ['motorbike', 'bike', 'motorcycle']:
            motorbike_class_id = key
        if value == 'person':
            person_class_id = key
        if value =='cow':
            cow_class_id = key

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_objects = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            # Skip detection if it's a person
            if class_id == person_class_id or class_id ==cow_class_id:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detected_objects.append((center_x, center_y, class_id))

        tracks = tracker.predict_and_update([(x, y) for x, y, _ in detected_objects])

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
            
            relative_velocity = max(np.mean([np.linalg.norm(other_kf.state[2:]) for other_id, other_kf in tracks.items() if other_id != track_id]), 0)
            ws.append([timestamp, velocity, relative_velocity])

            for (center_x, center_y, class_id), box in zip(detected_objects, results[0].boxes):
                if abs(pred_x - center_x) < 50 and abs(pred_y - center_y) < 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if class_id != motorbike_class_id else (0, 255, 0)
                    label = f'ID: {track_id}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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

    cap.release()
    out.release()
    wb.save(output_file_name)
    wb.close()

    print("Processing complete.", flush=True)
    full_output_file_path = os.path.abspath(output_file_name)
    subprocess.run([sys.executable, 'modelExcel.py', full_output_file_path])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        process_video(video_path)
    else:
        print("No video path provided.")
