# Blood Group Classification using CNN with Residual Blocks

This project implements a deep learning model to classify blood groups from images using a custom Convolutional Neural Network (CNN) with Residual Blocks. The project is built with PyTorch and includes a Streamlit web application for real-time inference and Grad-CAM visualization to interpret the model's predictions.

##  Features

- **Custom CNN Architecture**: A PyTorch-based CNN integrated with a Residual Block for improved feature extraction and gradient flow.
- **8-Class Classification**: Classifies images into 8 distinct blood groups: `A-`, `A+`, `B-`, `B+`, `AB-`, `AB+`, `O-`, `O+`.
- **Interactive Web App**: A Streamlit application (`app.py`) that allows users to upload images and get instant predictions.
- **Explainable AI (XAI)**: Integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of the image the model focuses on to make its predictions.
- **Jupyter Notebooks**: Includes notebooks for training (`train.ipynb`) and Grad-CAM experimentation (`gradcam_cnn.ipynb`).

##  Project Structure

```text
CNN/
│
├── app.py                                # Streamlit web application for inference & Grad-CAM
├── train_cnn.py                          # Python script for training the CNN model
├── train.ipynb                           # Jupyter Notebook for interactive training
├── gradcam_cnn.ipynb                     # Jupyter Notebook for Grad-CAM visualization
├── model/
│   └── best_basic_cnn_with_residual.pth  # Saved PyTorch model weights
├── Output/
│   └── model_summary.txt                 # Detailed model architecture and parameter summary
└── README.md                             # Project documentation
```

##  Model Architecture

The model is a custom CNN that takes `224x224` RGB images as input. It consists of:
- 4 Convolutional Layers with ReLU activation and Max Pooling.
- 1 Custom **Residual Block** (with 2 Convolutional layers and Batch Normalization) inserted after the 3rd convolutional layer to help with deep feature learning.
- A Fully Connected (Linear) classifier with Dropout (0.5) for regularization.

**Summary:**
- **Total Parameters**: 26,378,568
- **Model Size**: ~105.5 MB
- **Input Shape**: `[1, 3, 224, 224]`
- **Output Shape**: `[1, 8]` (8 blood group classes)

*(For a detailed layer-by-layer breakdown, see `Output/model_summary.txt`)*

##  Installation & Setup

1. **Clone the repository** (or download the project folder):
   ```bash
   git clone https://github.com/sujan22359/BloodGroup_identification_using_fingerprint.git
   cd BloodGroup_identification_using_fingerprint/CNN
   ```

2. **Create a virtual environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   Make sure you have PyTorch installed according to your system's CUDA availability. Then install the other requirements:
   ```bash
   pip install torch torchvision
   pip install streamlit numpy pillow matplotlib grad-cam tqdm
   ```

##  Usage

### 1. Running the Web Application
To start the Streamlit app for inference and Grad-CAM visualization:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`) where you can upload images and see the model's predictions along with the Grad-CAM heatmaps.

### 2. Training the Model
If you want to retrain the model on your own dataset:
1. Update the `data_dir` path in `train_cnn.py` to point to your dataset directory (which should be split into `train`, `val`, and `test` folders).
2. Run the training script:
   ```bash
   python train_cnn.py
   ```
   *Alternatively, you can use the `train.ipynb` notebook for an interactive training experience.*

### 3. Grad-CAM Experimentation
Open `gradcam_cnn.ipynb` in Jupyter Notebook or VS Code to experiment with the Grad-CAM visualizations step-by-step.

##  Technologies Used

- **Deep Learning Framework**: PyTorch, Torchvision
- **Web Framework**: Streamlit
- **Computer Vision & Image Processing**: PIL (Pillow), OpenCV (via grad-cam)
- **Explainable AI**: pytorch-grad-cam
- **Data Visualization**: Matplotlib
