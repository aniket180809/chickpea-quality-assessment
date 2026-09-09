# 🌱 Chickpea Quality Assessment

A computer vision project for automatically identifying and classifying chickpeas based on their visual quality and defects using deep learning.

## 📌 About the Project

Manual inspection of agricultural products can be time-consuming and subjective. This project explores an automated computer-vision approach for assessing chickpea quality from images.

The system is designed to identify different visual quality categories of chickpeas, including:

* ✅ Good
* 🟤 Shriveled
* 🟡 Discolored
* 🔴 Infected

The project uses deep learning and image-based classification/detection techniques to recognize quality characteristics from chickpea images.

The goal is to provide a foundation for an automated quality-inspection system that can assist with faster and more consistent agricultural product assessment.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Detect chickpeas from input images.
2. Identify visible quality defects.
3. Classify chickpeas into predefined quality categories.
4. Reduce dependence on manual visual inspection.
5. Explore the application of computer vision in agricultural quality assessment.

---

## 🧠 Approach

The project uses a deep-learning-based computer vision pipeline.

### Image Input

Images of chickpeas are provided as input to the system.

### Object Detection

A **YOLOv8** model is used to detect chickpeas within images.

The detection stage allows individual chickpeas to be localized using bounding boxes.

### Quality Classification

Detected/cropped chickpeas can then be analyzed according to their visual characteristics and assigned to the appropriate quality category.

The quality categories used in the dataset include:

```text
Good
Shriveled
Discolored
Infected
```

---

# 🛠️ Technologies Used

* Python
* YOLOv8
* Deep Learning
* Computer Vision
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Ultralytics
* Jupyter Notebook

---

# 📂 Project Structure

A typical project structure is:

```text
Chickpea-Quality-Assessment/
│
├── train/
│   ├── good/
│   ├── shriveled/
│   ├── discolored/
│   └── infected/
│
├── val/
│   ├── good/
│   ├── shriveled/
│   ├── discolored/
│   └── infected/
│
├── runs/
│   └── ...                    # Training results
│
├── best.pt                    # Best trained YOLO model
├── data.yaml                  # Dataset configuration
├── notebooks/                 # Analysis/training notebooks
├── README.md
└── requirements.txt
```

> The exact structure may vary depending on how the training dataset and notebooks are organized.

---

# 🚀 Installation

## 1. Install Python

Install Python 3.9 or later.

Verify your installation:

```bash
python --version
```

---

## 2. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd chickpea-quality-assessment
```

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

Install the required packages:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib jupyter
```

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Option 1 — Run the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the relevant notebook and execute the cells sequentially.

---

## Option 2 — Run YOLOv8 Inference

Once the trained model is available, an image can be passed to the YOLOv8 model for prediction.

Example:

```bash
yolo predict model=best.pt source="path/to/image.jpg"
```

For a directory of images:

```bash
yolo predict model=best.pt source="path/to/images/"
```

The prediction results will contain the detected objects and their predicted classes.

---

# 🏋️ Model Training

The YOLOv8 model was trained on the chickpea dataset to learn visual patterns associated with different quality categories.

Training can be performed using:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

### Parameters

| Parameter | Description                          |
| --------- | ------------------------------------ |
| `model`   | YOLOv8 model architecture/checkpoint |
| `data`    | Dataset configuration file           |
| `epochs`  | Number of training iterations        |
| `imgsz`   | Input image resolution               |

The trained model generates checkpoints and evaluation results in the YOLO training output directory.

---

# 📊 Dataset

The dataset contains images of chickpeas categorized according to their visual quality.

The primary categories include:

| Class      | Description                                            |
| ---------- | ------------------------------------------------------ |
| Good       | Chickpeas with acceptable visual quality               |
| Shriveled  | Chickpeas showing visible shrinkage or wrinkling       |
| Discolored | Chickpeas with abnormal coloration                     |
| Infected   | Chickpeas showing visible signs of infection or damage |

The dataset is divided into training and validation data to allow the model to learn from one set of images and evaluate its performance on unseen images.

---

# 🔬 Model Evaluation

Model performance can be evaluated using metrics commonly used for object detection and classification, including:

* Precision
* Recall
* mAP (mean Average Precision)
* Confusion Matrix
* Training/validation loss

YOLOv8 also provides visualization of training metrics and prediction results, which can be used to understand where the model performs well and where further training or data improvements may be required.

---

# 📈 Results

The trained YOLOv8 model provides predictions for chickpea images by identifying the location and quality category of detected chickpeas.

The training results, validation metrics, prediction samples, and model performance visualizations are available in the generated training outputs.

> Model performance numbers should be added here after final evaluation so that the README reports the actual measured results rather than estimated values.

---

# 💡 Applications

This type of system could be useful for:

* Agricultural product quality inspection
* Automated sorting systems
* Food-processing quality control
* Seed quality assessment
* Agricultural research
* Computer-vision-based grading systems

---

# 🔮 Future Improvements

Potential improvements include:

* Increasing the size and diversity of the dataset
* Collecting images under different lighting conditions
* Improving class balance
* Increasing the number of training examples for rare defects
* Comparing different YOLOv8 model sizes
* Hyperparameter tuning
* Adding real-time camera-based detection
* Building an automated chickpea sorting pipeline
* Deploying the model as a web application or API
* Combining object detection with a dedicated quality classifier

---

# 📁 Important Files

### `best.pt`

The trained YOLOv8 model checkpoint containing the learned model weights.

### `data.yaml`

Defines the dataset locations and class names used during YOLO training.

### `train/`

Contains training images and annotations/classes.

### `val/`

Contains validation images used to evaluate model performance during training.

---

# ⚠️ Limitations

The quality assessment system is dependent on the quality and diversity of the training data.

Factors such as:

* Lighting
* Camera quality
* Image background
* Chickpea orientation
* Defect visibility
* Dataset size
* Class imbalance

can affect prediction performance.

Therefore, additional real-world images and validation are required before using the system in an actual industrial quality-control environment.

---

# 👨‍💻 Author

**Aniket Raikar**

GitHub:
https://github.com/aniket180809

---

## ⭐ Project Summary

This project demonstrates the application of **deep learning and computer vision to agricultural quality assessment**, using YOLOv8 to detect and classify chickpeas according to their visible quality characteristics.

It combines image processing, dataset preparation, deep-learning model training, object detection, and model evaluation to explore an automated approach to chickpea quality inspection.
