# 🧠 Custom VGG16 Implementation with PyTorch Components Comparison

## 📌 Overview
This project presents a complete **from-scratch implementation of VGG16** using manually built deep learning components in PyTorch. The objective is to understand and compare the performance of **custom modules** like BatchNorm, ReLU, MaxPooling, and Dropout against their **PyTorch built-in counterparts**.

---

## 📁 Project Structure

| Section              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Custom Components`  | Manually implemented layers for BatchNorm2d, ReLU, MaxPooling2d, Dropout   |
| `Custom VGG16`       | Full VGG16-like model using custom components for MNIST classification      |
| `Baseline VGG16`     | Standard VGG16 implementation using PyTorch built-in layers                 |
| `Trainer`            | Training loop with accuracy/loss monitoring for both models                 |
| `Evaluator`          | Model evaluation on test dataset                                            |

---

## 🛠️ Technologies Used

- **PyTorch**
- **NumPy**
- **Matplotlib** (for plotting accuracy/loss curves)
- **MNIST Dataset**
- **Google Colab / CPU/GPU**

---

## 🔍 Custom Components Implemented

- `CustomBatchNorm2d`: Normalizes across batch and updates running statistics
- `CustomReLU`: Element-wise activation using `torch.max`
- `CustomMaxPooling2d`: Patch-wise max pooling using `F.unfold` and `view`
- `CustomDropout`: Random element dropout during training

These were integrated into a custom VGG16 model for experimental evaluation.

---

## 🧠 VGG16 Architecture

- 13 Convolutional Layers in 5 blocks
- 3 Fully Connected Layers
- Each block uses: Conv → CustomBN → CustomReLU → (Optional: CustomMaxPooling2d)
- Final FC layers adapted for MNIST (10-class classification)

---

## 🧪 Experiment Setup

### Dataset
- **MNIST** (handwritten digit classification)
- Image size: Up-scaled to 224x224 to fit VGG16 input format

### Training Parameters
- **Epochs**: 3  
- **Batch Size**: 16  
- **Optimizer**: Adam  
- **Learning Rate**: 0.001  

### Evaluation Criteria
- Training Accuracy & Loss
- Test Accuracy & Loss
- Learning Curves


---

## 📚 Future Improvements

- Add learning rate scheduling
- Handle edge cases in pooling more robustly
- Optimize custom modules for speed & memory
- Introduce data augmentation for MNIST
- Extend support for arbitrary input sizes

---
