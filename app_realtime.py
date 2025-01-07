import streamlit as st
import cv2
import os
import numpy as np
import torch
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, MODEL_PATH, OUTPUT_PATH
from prediction_utils import  predict_with_model


# Realtime
def process_rtsp_stream(yolo_model, resnet_model, rtsp_url, skip_frames, stop_signal):
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0

    if not cap.isOpened():
        st.error(f"Error: Cannot open RTSP stream: {rtsp_url}")
        return

    st_frame = st.empty()  #  video frames
    results_board = st.empty()  #  results board

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame from RTSP stream.")
            break

        # Stop the stream button 
        if stop_signal():
            st.info("Stream stopped.")
            break
        
        device = 0 if torch.cuda.is_available() else 'cpu'

        # YOLO detection
        detections = yolo_model.predict(frame,device=device, conf=0.4) #device=0 chạy gpu

        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the box

        #  `skip_frames`=> crop ảnh => classify từng crop
        if frame_count % skip_frames == 0:
            healthy_count = 0  # Reset class healthy cho cập nhật prediction tiếp theo
            sick_count = 0     # Reset class sick cho cập nhật prediction tiếp theo

            for i, det in enumerate(detections[0].boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cropped = frame[y1:y2, x1:x2]  # crop ảnh theo tọa độ pixel 

                if cropped.size == 0:  
                    continue

                # Save cropped image 
                os.makedirs(OUTPUT_PATH, exist_ok=True)  # Ensure the directory exists
                crop_path = os.path.join(OUTPUT_PATH, f"frame_{frame_count}_crop_{i}.jpg")
                cv2.imwrite(crop_path, cropped)

                # Predict ảnh crop
                predicted_class, confidence_healthy, confidence_sick = predict_with_model(resnet_model, crop_path)

                # Update 
                if predicted_class == "Healthy":
                    healthy_count += 1
                else:
                    sick_count += 1

                # Delete the cropped image sau cho lần kế tiếp
                if os.path.exists(crop_path):
                    os.remove(crop_path)

            # Update the results board sau khi dự đoán ảnh cuối
            results_board.markdown(
                f"### Results Board\n"
                f"- **Healthy Pigs**: {healthy_count}\n"
                f"- **Sick Pigs**: {sick_count}"
            )

        # Display Streamlit
        st_frame.image(frame, channels="BGR", use_container_width=True)
        frame_count += 1

    cap.release()
    st.info("Stream ended.")

# Streamlit app realtime
def main():
    st.title("Health Classification Realtime")
    st.write("Provide an RTSP stream URL for real-time object detection and classification.")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    resnet_model = load_model(MODEL_PATH)

    # Input RTSP stream URL
    rtsp_url = st.text_input("Enter RTSP Stream URL:", "rtsp://admin:farm1234@minh1234.smartddns.tv:5540/cam/realmonitor?channel=1&subtype=0")
    skip_frames = st.number_input("Enter the number of frames to skip for classification:", min_value=1, max_value=100, value=10)

    #  start and stop the stream
    if st.button("Start Processing Stream"):
        stop_button = st.button("Stop Stream")

        # Define a function to monitor the stop button
        def stop_signal():
            return stop_button

        st.write("Processing RTSP stream...")
        process_rtsp_stream(yolo_model, resnet_model, rtsp_url, skip_frames, stop_signal)

if __name__ == "__main__":
    main()
