import os
import glob
import cv2
import numpy as np
from skimage.measure import label
from sklearn.neural_network import MLPClassifier

# Use the instructor's background extraction strategy
def extract_foreground_mask(img, kernel_size=9):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    _, img_bw = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    labels = label(img_bw_cleaned)
    if len(img_bw_cleaned.flat) == 0 or np.sum(img_bw_cleaned) == 0:
        return np.zeros_like(img_gray)
    label_of_largest_region = np.argmax(np.bincount(labels.flat, weights=img_bw_cleaned.flat))
    return (labels == label_of_largest_region).astype(np.uint8)

def compute_features(image):
    mask = extract_foreground_mask(image)
    area = float(np.sum(mask))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    else:
        aspect_ratio = 1.0
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1]
    mean_saturation = float(cv2.mean(saturation_channel, mask=mask)[0])
    
    return np.array([area, aspect_ratio, mean_saturation])

if __name__ == "__main__":
    X_list, y_list = [], []
    
    print("Reading data...")
    for f in (glob.glob("not_olive_fly/*.JPG") + glob.glob("not_olive_fly/*.jpg")):
        img = cv2.imread(f)
        if img is not None: X_list.append(compute_features(img)); y_list.append(0)
            
    for f in (glob.glob("olive_fly/*.JPG") + glob.glob("olive_fly/*.jpg")):
        img = cv2.imread(f)
        if img is not None: X_list.append(compute_features(img)); y_list.append(1)
            
    X, y = np.array(X_list), np.array(y_list)
    
    # Feature Standardization Scaling
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std == 0] = 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # Define a tiny, highly efficient neural net: 3 inputs -> 4 hidden neurons -> 1 output
    mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, random_state=42)
    mlp.fit(X_scaled, y)
    
    print("\n=======================================================")
    print("🎉 NEURAL NETWORK TRAINING SUCCESSFUL! COPY THESE PARAMETERS:")
    print("=======================================================")
    print(f"X_MEAN = np.array({list(X_mean)})")
    print(f"X_STD  = np.array({list(X_std)})")
    print(f"W1 = np.array({mlp.coefs_[0].tolist()})")
    print(f"B1 = np.array({mlp.intercepts_[0].tolist()})")
    print(f"W2 = np.array({mlp.coefs_[1].tolist()})")
    print(f"B2 = np.array({mlp.intercepts_[1].tolist()})")
    print("=======================================================")