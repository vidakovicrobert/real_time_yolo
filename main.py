from ultralytics import YOLO
import cvzone
import cv2

# Load your custom trained model
model = YOLO('best.pt')  # Change here to use your custom trained model

confidence_threshold = 0.5


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to 0 or 1 based on your webcam index
if not cap.isOpened():
    print("Failed to open camera!")
else:
    print("Camera opened successfully!")

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

    # Show the image with detection results
    cv2.imshow('frame', image)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
