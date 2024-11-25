# U2Net Image Segmentation with TensorFlow

This project demonstrates how to use the **U2Net architecture** for semantic image segmentation using TensorFlow.
![Architecture](architecture.jpg "This is a image for U2NET architecture")

## Key Features

- **U2Net Architecture**: A deep learning model for semantic segmentation.
- **GPU Optimization**: Takes advantage of CUDA and TensorFlow's mixed precision training.
- **Easy Dataset Handling**: Automatically processes image-mask pairs for training and validation.
- **Callbacks**: Includes model saving, learning rate scheduling, early stopping, and TensorBoard logging.
- **Dynamic Memory Growth**: Prevents TensorFlow from pre-allocating excessive GPU memory.

---

## 📁 Project Structure

```plaintext
.
├── images/              # Directory for input images
├── masks/               # Directory for ground truth masks
├── model/               # Implementation of U2Net architecture
├── files/               # Directory for saved model and logs
├── logs/                # TensorBoard logs for visualization
└── train.py              # Main script to train and evaluate the model
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/u2net-segmentation.git
   cd u2net-segmentation
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ How to Use

1. Prepare your dataset:
   - Place input images in the `images/` directory.
   - Place corresponding masks in the `masks/` directory.
2. Start training:
   ```bash
   python train.py
   ```

---

## 🔧 Customization

- **Adjust Hyperparameters**: Modify these values in `main.py`:
  ```python
  BATCH_SIZE = 4
  LEARNING_RATE = 1e-4
  EPOCHS = 10
  ```
- **Dataset Split**: Update the `split` parameter in the `load_dataset()` function to change the train-validation ratio.
- **Model Selection**: Switch between `build_u2net()` and `build_u2net_lite()` in the script for different variants of the U2Net architecture.

---

## 🏁 Outputs

- **Saved Model**: Stored in the `files/` directory as `model.keras`.
- **Training Logs**:
  - Loss and validation logs: `files/log.csv`.

---

## 📊 Visualizing Results

To view the training and validation loss curves:

1. Run the training script. After training completes, the loss plot will be displayed automatically.

---
