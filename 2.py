import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8m_custom.pt')

# Load the image using a raw string for the file path
image = cv2.imread(r'D:\shivu\people\runs\detect\predict5\p4.jpg')

# Check if the image is loaded properly
if image is None:
    raise FileNotFoundError("Image not found or unable to load.")

# Run inference
results = model(image)

# Get the results from the first (and only) item in the results list
boxes = results[0].boxes  # Box objects
xyxy = boxes.xyxy.cpu().numpy()  # Extract the bounding box coordinates
conf = boxes.conf.cpu().numpy()  # Extract the confidence scores
cls = boxes.cls.cpu().numpy()  # Extract the class IDs

# Draw bounding boxes on the image
for box in xyxy:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Count the number of people detected
num_people = len(xyxy)

# Display the result
cv2.putText(image, f'Number of people: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow('people Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of people detected
print(f'Number of people detected: {num_people}')