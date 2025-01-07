import streamlit as st
import cv2
import os
import numpy as np
import torch
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, MODEL_PATH, OUTPUT_PATH
import tempfile
from prediction_utils import  predict_with_model


#Video
def process_video(yolo_model, resnet_model, video_path, output_path, skip_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Get video 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # bảo đảm file tồn tạo hoặc tạo mới
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # tạo copy trc khi crop
        frame_without_boxes = frame.copy()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # YOLO detection
        detections = yolo_model.predict(frame,device=device, conf=0.2) #device=0 chạy gpu

        # vẽ YOLO bounding boxes 
        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the box

        # Rết class cho new prediction
        if frame_count % skip_frames == 0:
            healthy_count = 0  # Reset class healthy cho cập nhật prediction tiếp theo
            sick_count = 0     # Reset class sick cho cập nhật prediction tiếp theo

            for i, det in enumerate(detections[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cropped = frame_without_boxes[y1:y2, x1:x2]  # crop ảnh theo tọa độ pixel 


                if cropped.size == 0:  
                    continue

                # Save cropped image 
                crop_path = os.path.join(OUTPUT_PATH, f"frame_{frame_count}_crop_{i}.jpg")
                cv2.imwrite(crop_path, cropped)

                # Predict with model
                predicted_class, confidence_healthy, confidence_sick = predict_with_model(resnet_model, crop_path)

                # Update 
                if predicted_class == "Healthy":
                    healthy_count += 1
                else:
                    sick_count += 1

                # Delete the cropped image sau khi predict 
                if os.path.exists(crop_path):
                    os.remove(crop_path)

        # Update the results board sau khi dự đoán ảnh cuối
        board_text = f"Healthy: {healthy_count} | Sick: {sick_count}"
        cv2.putText(frame, board_text, (width - 800, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)


        # combind frame lại thành video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


# Streamlit app for video processing
def main():
    st.title("Video heallth classification")
    st.write("Upload a video file for object detection and classification.")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    resnet_model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    skip_frames = st.number_input("Enter the number of frames to skip for classification:", min_value=1, max_value=100, value=10)

    if uploaded_file is not None:
        # Save the uploaded video 
        video_path = os.path.join(OUTPUT_PATH, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Save video path
        output_path = os.path.join(OUTPUT_PATH, "processed_video.mp4")

        # Process the video
        st.write("Processing video...")
        process_video(yolo_model, resnet_model, video_path, output_path, skip_frames)
        st.success("Processing completed!")
        st.write(f"Video saved to: {output_path}")

        # Display the processed video
        if os.path.exists(output_path):
            st.video(output_path)
            
        else:
            st.error("Error: Processed video was not saved correctly.")

if __name__ == "__main__":
    main()
