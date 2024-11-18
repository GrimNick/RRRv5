import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time
from openpyxl import Workbook
import sys

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
        self.Q = np.eye(4) * 0.1
        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 5
        self.P = np.eye(4) * 100

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

class KalmanVehicleTracker:
    def __init__(self, memory_limit=30):
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

            for detection in unmatched_detections:
                distance = np.linalg.norm(np.array(detection) - predicted_pos)
                if distance < best_distance and distance < 50:
                    best_distance = distance
                    best_match = detection

            if best_match is not None:
                kf.update(best_match)
                updated_tracks[track_id] = kf
                unmatched_detections.remove(best_match)
                
                velocity = np.linalg.norm(kf.state[2:])
                if track_id not in self.velocity_history:
                    self.velocity_history[track_id] = []
                self.velocity_history[track_id].append(velocity)

        for detection in unmatched_detections:
            reassociated = False
            for track_id, (kf, age) in self.lost_tracks.items():
                if age < self.memory_limit:
                    predicted_pos = kf.predict()
                    distance = np.linalg.norm(np.array(detection) - predicted_pos)
                    if distance < 50:
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

tracker = KalmanVehicleTracker(memory_limit=60)

def format_timestamp(frame_count, fps):
    total_seconds = frame_count / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_video(video_path):
    print(f"Processing video at: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # Use H.264 codec for better compatibility with HTML5
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    output_file_name = output_path.replace('.mp4', '_data.xlsx')
    wb = Workbook()
    frame_count = 0
    
    motorbike_class_id = None
    for key, value in model.names.items():
        if value in ['motorbike', 'bike','motorcycle']:
            motorbike_class_id = key
            break

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_objects = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detected_objects.append((center_x, center_y, class_id))

        tracks = tracker.predict_and_update([(x, y) for x, y, _ in detected_objects])

        timestamp = format_timestamp(frame_count, fps)

        for track_id, kf in tracks.items():
            # Extract predicted position (x, y)
            pred_x, pred_y = kf.state[:2]

            # Calculate the velocity from the Kalman filter state (assuming state [x, y, vx, vy])
            velocity = np.linalg.norm(kf.state[2:])

            # Subtract 0.5 for noise calculation
            velocity -= 0.5

            # Reset velocity to 0 if it is below 0
            if velocity < 0:
                velocity = 0

            relative_velocity = 0

            # Compute relative velocity if there are other tracks
            if len(tracks) > 1:
                velocities = [np.linalg.norm(other_kf.state[2:]) for other_id, other_kf in tracks.items() if other_id != track_id]
                relative_velocity = np.mean(velocities) if velocities else 0

            # Define the sheet name for the current track ID
            sheet_name = f'Track ID {track_id}'

            # Check if the sheet already exists; if not, create it
            if sheet_name not in wb.sheetnames:
                ws = wb.create_sheet(sheet_name)
                ws.append(["Time", "Velocity", "Relative Velocity"])
            else:
                ws = wb[sheet_name]

            # Append the current timestamp, velocity, and relative velocity to the sheet
            ws.append([timestamp, velocity, relative_velocity])

            # Drawing bounding boxes and labels
            for (center_x, center_y, class_id), box in zip(detected_objects, results[0].boxes):
                if abs(pred_x - center_x) < 50 and abs(pred_y - center_y) < 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0) if class_id != motorbike_class_id else (0, 255, 0)
                    label = f'ID: {track_id}'

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    wb.save(output_file_name)
    wb.close()
   

    # Print message to indicate completion
    print("Processing complete", flush=True)  # This will output to stdout


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]  # Get the video path from the command line arguments
        process_video(video_path)
    else:
        print("No video path provided.")


# Set the video path and process the video
video_path = 'E:/Videoo/track2.mp4'  # Replace with your video path
process_video(video_path)
