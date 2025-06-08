# Glaucoma-Detection

#DATASET
https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2

# Glaucoma Detection Using Deep Learning

This project aims to detect glaucoma from retinal fundus images using deep learning. Leveraging transfer learning with state-of-the-art convolutional neural networks (CNNs), the pipeline enhances model accuracy through data augmentation and class balancing.

## 📁 Project Structure

- `Glaucoma Detection.ipynb`: Jupyter Notebook implementing the full pipeline.
- 
- `README.md`: Project overview and setup instructions.

## 🧠 Models Used

1. **MobileNetV2**
   - Lightweight and efficient
   - Ideal for mobile and edge deployment
   - Fine-tuned with custom dense layers

2. **DenseNet**
   - Dense connections to encourage feature reuse
   - Reduces the vanishing gradient problem
   - Fewer parameters than traditional deep CNNs

3. **Xception**
   - Uses depthwise separable convolutions
   - Performs better on high-resolution images
   - Advanced architecture based on Inception modules

## 🔄 Pipeline Overview

1. **Data Loading**
   - Fundus images with glaucoma labels

2. **Preprocessing**
   - Resize and normalize images
   - Augmentation: rotation, flip, zoom, brightness adjustment
   - Class balancing (oversampling minority class)

3. **Model Building**
   - Load pretrained base model (MobileNetV2, DenseNet, or Xception)
   - Add GlobalAveragePooling, Dropout, and Dense layers
   - Compile with Adam optimizer and categorical crossentropy

4. **Training & Evaluation**
   - Visualize training history (accuracy/loss curves)
   - Use validation split to monitor overfitting
   - Generate classification metrics (e.g., confusion matrix, precision, recall)

## 📊 Results (To Be Updated)

| Model       | Validation Accuracy | Notes                        |
|-------------|---------------------|------------------------------|
| MobileNetV2 | `0.8455`               | Lightweight and fast         |
| DenseNet    | `0.8195`               | Better generalization        |
| Xception    | `0.8623`               | High accuracy with large input |

## 📦 Requirements

- Python 3.7+
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
