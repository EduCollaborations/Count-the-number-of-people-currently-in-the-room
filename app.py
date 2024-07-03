import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model and the prototxt file
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt', 
    'mobilenet_iter_73000.caffemodel'
)

# List of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Open a video file or an image file or a webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with 'path/to/video' for a video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get the height and width of the frame
    (h, w) = frame.shape[:2]

    # Preprocess the frame: resize to 300x300 pixels and normalize it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Initialize a counter for people
    person_count = 0

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # If the class label is "person", increment the person count
            if CLASSES[idx] == "person":
                person_count += 1

                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
