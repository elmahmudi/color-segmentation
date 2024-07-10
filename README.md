# Image Segmentation Using Detectron2 and YOLOv8

This project demonstrates image segmentation using two different frameworks: Detectron2 and YOLOv8. It covers steps from dataset
preparation to model training, inference, and evaluation.

# Build image segmentation using Detectron2 
You need to follow the instructions provided in Detectron2 Segmentation.ipynb
1. **Install Detectron2**

2. **Create Annotations**:
    - Annotate your images in COCO JSON format using tools like [MakeSense.ai](https://www.makesense.ai/).
    
3. **Set Directories**:
    - Define paths for training and validation images and their respective JSON annotation files.
4. **Configure and Train Model**:
    - Configure the model settings.
    - Initiate the training process.

5. **Inference and Evaluation**:
    - Load the trained model for inference.
    - Visualize the segmentation results.
---

# Build image segmentation using Yolo v8 

You need to follow the instructions provided in YOLO V8 Segmentation.ipynb

1. **Prepare Dataset**:
    - Annotate your images using tools like [MakeSense.ai](https://www.makesense.ai/) or [Roboflow](https://roboflow.com/) and convert annotations to YOLO format.

2. **Install Required Libraries**:
    - Install the `ultralytics` package and other necessary libraries.

3. **Import Libraries**:
    - Import necessary libraries including YOLOv8 and GPU utilities.

4. **Model Setup**:
    - Load pre-trained YOLOv8 models.
    - Configure the paths for your dataset and results.
5. **Train Model**:
    - Configure training settings.
    - Train your model using the uploaded dataset.

6. **Inference and Evaluation**:
    - Load the trained model for inference.
    - Perform segmentation on new images and visualize the results.

7. **Predeict masks for images**:
    - Folder of multiple images.
    - Extract masks and save them.