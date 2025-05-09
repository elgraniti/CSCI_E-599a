# How to Train a YOLO Model with Ultralytics

This guide explains how to set up and train a YOLO model using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework in Python.

---

## Dataset Configuration (`data.yaml`)

The `data.yaml` file defines the dataset structure and class labels. It must be located in the same directory as your training script or notebook.

```yaml
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ["class_0", "class_1"]




Environment Setup
Install Ultralytics YOLO:
In Bash:

pip install ultralytics
pip install pandas 

Please add all the necessary libraries.

Running the notebook
The notebook is attached. Below is a version of the notebook.

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  # Should be True
from ultralytics import YOLO
import torch

# Clear cache if needed
torch.cuda.empty_cache()

# Load the model (choose one)
# model = YOLO("yolov8n.pt")  # nano
# model = YOLO("yolov8m.pt")  # medium
model = YOLO("yolov8x.pt")    # extra large

# Train the model
results = model.train(
    data="data.yaml",
    epochs=500,
    imgsz=1280,
    batch=16,
    device=[0, 1],             # Multi-GPU training
    cache="disk",              # Use RAM or disk for caching
    augment=True,
    mosaic=1.0,
    mixup=0.5,
    amp=True,                  # Mixed precision
    conf=0.05,                 # Lower confidence threshold to improve recall
    lr0=0.001,
    lrf=0.0001
)


Validating or Testing (Optional)

metrics = model.val()

results = model.predict(source='images/test')


