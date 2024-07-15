## Counting the Number of People Using OpenCV and YOLOv8

### Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Software and Tools](#software-and-tools)
- [Dataset Preparation](#dataset-preparation)
- [Configuration File](#configuration-file)
- [Model Setup](#model-setup)
- [Training the Model](#training-the-model)
- [Rename Trained Weights](#rename-trained-weights)
- [Model Inference](#model-inference)
- [Explanation Code](#explanation-code)
- [Python Code for People Detection](#python-code-for-people-detection)
- [Output Explanation](#output-explanation)
- [Applications](#applications)
- [Future Scope](#future-scope)
- [Summary](#summary)

### Introduction
This project involves annotating a dataset containing images with people, training a YOLOv8 model for counting the number of people in a room, and using the trained model to detect the number of people present in the room. The steps below detail the entire process from dataset preparation to model training and testing.

### Requirements
- Python 3.x
- OpenCV 4.x

### Software and Tools
- **Python**: Programming language used for the script.
- **OpenCV**: Library for computer vision tasks.

### Dataset Preparation
1. **Annotate Images**: Use a tool like labelImg to annotate the dataset containing images with bounding boxes around people. Save the annotations in the appropriate YOLO format.
2. **Split Dataset**: Split the annotated dataset into training and validation sets, each containing images and labels folders.

### Configuration File
Create a `data.yaml` file with the following content:
```yaml
train: D:\shivu\people\train\images
val: D:\shivu\people\valid\images
nc: 1
names: ["people"]
```

### Model Setup
1. **Install Dependencies**:
    ```bash
    pip install ultralytics opencv-python opencv-contrib-python
    ```
2. **Activate Virtual Environment**:
    ```bash
    python -m venv myenv
    cd myenv
    cd Scripts
    activate
    ```

3. **Verify Installation**:
    ```bash
    python -c "import torch; print(torch.__version__)"
    python -c "import torch; print(torch.cuda.is_available())"
    ```

### Training the Model
Train the YOLOv8 model using the following script (`1.py`):
```python
import ultralytics
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(data="D:\shivu\people\data.yaml", epochs=3)
```
This process might take some time. Upon completion, you will get a `best.pt` file located inside `runs/detect/train/weights`.

### Rename Trained Weights
Rename the `best.pt` file to `yolov8m_custom.pt` and move it to the root directory.

### Model Inference
To detect people in an image using the trained model, use the following command:
```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=path/to/your/image.jpg
```

### Explanation Code

### Python Code for People Detection
Use the following Python code (`2.py`) to import the trained model `yolov8m_custom.pt` and detect people in an image, displaying the number of people detected:
```python
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
cv2.imshow('People Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of people detected
print(f'Number of people detected: {num_people}')
```

### Output Explanation
After running the inference code, the output will be an image with detected people highlighted by green rectangles. The number of detected people will be displayed both on the image and printed in the console.

### Applications
The techniques and processes described in this project have several practical applications, including:
- **Automated Attendance**: Counting the number of people in classrooms or meetings.
- **Surveillance and Security**: Monitoring and detecting people in security footage.
- **Event Management**: Estimating the number of attendees in an event.

### Future Scope
Future enhancements and extensions to this project could include:
- **Multi-Class Detection**: Expanding the model to detect and classify multiple types of objects.
- **Real-Time Detection**: Implementing real-time detection using video feeds.
- **Improved Accuracy**: Fine-tuning the model and using more sophisticated data augmentation techniques to improve detection accuracy.
- **Deployment**: Creating a web or mobile application to deploy the model for practical use.

### Summary
This project demonstrates the complete pipeline of annotating a dataset, training a YOLOv8 model, and using the trained model to detect people in images. By following the steps outlined, one can develop a custom object detection model tailored to specific needs, with various practical applications across different industries. The future scope suggests further improvements and extensions to enhance the model's capabilities and deployment options.

---
