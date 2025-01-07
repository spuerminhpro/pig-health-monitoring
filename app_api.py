import streamlit as st
import cv2
import os
import json
import numpy as np
import torch
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, MODEL_PATH, OUTPUT_PATH
from prediction_utils import predict_with_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Video processing function
def process_video(yolo_model, resnet_model, video_path, skip_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []  # Store frame-level results

    if not cap.isOpened():
        return {"error": f"Could not open video file: {video_path}"}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        device = 0 if torch.cuda.is_available() else 'cpu'

        # YOLO detection 
        detections = yolo_model.predict(frame, device=device, conf=0.4)

        if frame_count % skip_frames == 0:
            healthy_count = 0
            sick_count = 0

            for i, det in enumerate(detections[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0 or cropped.shape[0] < 32 or cropped.shape[1] < 32:
                    print(f"Invalid crop in frame {frame_count}, skipping...")
                    continue

                # Save cropped image 
                crop_path = os.path.join(OUTPUT_PATH, f"frame_{frame_count}_crop_{i}.jpg")
                cv2.imwrite(crop_path, cropped)

                # Classification
                predicted_class, confidence_healthy, confidence_sick = predict_with_model(resnet_model, crop_path)

                if predicted_class == "Healthy":
                    healthy_count += 1
                else:
                    sick_count += 1

                if os.path.exists(crop_path):
                    os.remove(crop_path)

            overall_status = "All pigs healthy" if sick_count == 0 else "Suspect pig"
            st.write({
                "frame_id": frame_count,
                "overall_status": overall_status
            })
            results.append({
                        "frame_id": frame_count,
                        "overall_status": overall_status
                    })

        frame_count += 1

    cap.release()


    return {
        "overall_status": overall_status
    }

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Video</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
            form { margin: auto; width: 300px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
            input[type="file"] { margin-bottom: 15px; }
            button { padding: 10px 20px; font-size: 16px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #218838; }
        </style>
    </head>
    <body>
        <h1>Upload Video for Processing</h1>
        <form action="/process_video" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required><br>
            <label for="skip_frames">Skip Frames:</label>
            <input type="number" name="skip_frames" value="10" min="1" max="100"><br><br>
            <button type="submit">Upload and Process</button>
        </form>
    </body>
    </html>
    '''


# Flask API 
@app.route('/process_video', methods=['POST'])
def api_process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_video_path = os.path.join(OUTPUT_PATH, file.filename)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    file.save(temp_video_path)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    resnet_model = load_model(MODEL_PATH)

    skip_frames = int(request.form.get('skip_frames', 90))
    results = process_video(yolo_model, resnet_model, temp_video_path, skip_frames)

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        
    # lưu kết quả vao file
    json_file_path = os.path.join(OUTPUT_PATH, "results.json")
    if os.path.exists(json_file_path):
        os.remove(json_file_path)  
    with open(json_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
        
    return jsonify(results)





# Streamlit 
def main():
    st.title("Video Health Classification")
    st.write("Upload a video file for object detection and classification.")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    resnet_model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    skip_frames = st.number_input("Enter the number of frames to skip for classification:", min_value=1, max_value=100, value=10)

    if uploaded_file is not None:
        video_path = os.path.join(OUTPUT_PATH, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("Processing video...")
        results = process_video(yolo_model, resnet_model, video_path, skip_frames)
        st.success("Processing completed!")

        # Display final JSON results
        st.json(results)


if __name__ == "__main__":
    #  Streamlit
    #main()
    # For API Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
