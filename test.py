import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf

H = 256
W = 256

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create necessary directories
    for item in ["joint", "mask"]:
        create_dir(f"results/{item}")

    # Load the model
    model_path = os.path.join("files", "model.h5")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        exit()

    model = tf.keras.models.load_model(model_path)

    # Load test images
    images = glob("tests/*")
    print(f"Images: {len(images)}")

    # Predict and save results
    for x in tqdm(images, total=len(images)):
        # Extract the name of the image
        name = os.path.basename(x)

        # Read and preprocess the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image {x}")
            continue

        x_resized = cv2.resize(image, (W, H))
        x_normalized = x_resized / 255.0
        x_input = np.expand_dims(x_normalized, axis=0)

        # Make the prediction
        pred = model.predict(x_input, verbose=0)

        line = np.ones((H, 10, 3)) * 255

        # Prepare predicted mask
        pred_list = []
        for item in pred:
            p = item[0] * 255
            p = np.concatenate([p, p, p], axis=-1)
            pred_list.append(p)
            pred_list.append(line)

        # Save the mask image
        save_mask_path = os.path.join("results", "mask", name)
        cat_images = np.concatenate(pred_list, axis=1)
        cv2.imwrite(save_mask_path, cat_images)

        # Save joint image (original + mask + masked image)
        image_h, image_w, _ = image.shape
        y0 = pred[0][0]
        y0_resized = cv2.resize(y0, (image_w, image_h))
        y0_resized = np.expand_dims(y0_resized, axis=-1)
        y0_resized = np.concatenate([y0_resized, y0_resized, y0_resized], axis=-1)

        line = np.ones((image_h, 10, 3)) * 255
        cat_images_joint = np.concatenate([image, line, y0_resized * 255, line, image * y0_resized], axis=1)
        save_joint_path = os.path.join("results", "joint", name)
        cv2.imwrite(save_joint_path, cat_images_joint)
