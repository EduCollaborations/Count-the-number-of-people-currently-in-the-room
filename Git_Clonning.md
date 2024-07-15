## Run Instructions

### Step 1: Get Requirements
Ensure you have Python, OpenCV, and the necessary YOLOv8 dependencies installed:
```bash
pip install ultralytics opencv-python opencv-contrib-python
```

### Step 2: Run Training Script
Execute the training script (`1.py`) to train the YOLOv8 model:
```python
python 1.py
```

### Step 3: Predict Using Trained Model
Run the following command to detect people in an image:
```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=path/to/your/image.jpg
```

### Step 4: Run Inference Script
Modify the image path in `2.py` to the desired image from the predictions folder and run the script:
```python
python 2.py
```

This will load the image, run inference using the trained model, and display the number of people detected in the image.
