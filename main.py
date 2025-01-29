from ultralytics import YOLO
import cvzone
import cv2

# Load your custom trained model
model = YOLO('best.pt')

# Set your desired confidence threshold (e.g., 50% confidence)
confidence_threshold = 0.4

# Try DirectShow for webcam compatibility
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change to 0 or 1 based on your webcam index
if not cap.isOpened():
    print("Failed to open camera!")
else:
    print("Camera opened successfully!")

# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up the VideoWriter to save the output video
output_filename = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec if needed (e.g., 'MJPG', 'MP4V')
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, image = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform detection
    results = model(image)
    for info in results:
        parameters = info.boxes
        for box in parameters:
            confidence = box.conf[0].numpy()
            
            if confidence < confidence_threshold:  # Filter detections below the threshold
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            class_detected_number = box.cls[0]
            class_detected_number = int(class_detected_number)
            class_detected_name = results[0].names[class_detected_number]

            # Draw bounding box and text on the frame
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cvzone.putTextRect(image, f'{class_detected_name} ({int(confidence * 100)}%)', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

    # Resize the image to a desired size (e.g., 640x480)
    image_resized = cv2.resize(image, (640, 480))
    # Show the image with detection results
    cv2.imshow('frame', image_resized)

    # Write the processed frame to the output video
    out.write(image)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
