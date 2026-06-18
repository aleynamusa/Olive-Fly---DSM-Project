import glob
import argparse
import pathlib
import cv2
import numpy as np
from skimage.measure import label

parser = argparse.ArgumentParser(
    prog="OliveFly detection test script",
    description="""
        This scripts tests the olive fly detection algorithm
        against a set of images.""")

parser.add_argument('directory', help='location of the dataset',
                    type=pathlib.Path)
parser.add_argument('--verbose', '-v', action="store_true")


# =======================================================
# YOUR MATHEMATICALLY OPTIMIZED NEURAL NETWORK PARAMETERS
# =======================================================
# These are your real parameters from cell 72 of your notebook!
X_MEAN = np.array([3877.45872340, 1.11746390, 149.34492621])
X_STD  = np.array([4801.63556521, 0.55243511, 51.88207313])

W1 = np.array([[-1.47879269,  0.95968934,  1.09317243, -1.74546277],
               [-0.26691265, -1.81149109,  0.42032040,  0.38288509],
               [-0.67466904,  0.17595713, -1.53510977,  1.35100705]])
B1 = np.array([0.68760984, -0.80685428,  0.79454286, -1.15430438])

W2 = np.array([[-0.81486777],
               [-0.93495785],
               [-0.85189586],
               [-1.67806577]])
B2 = np.array([0.24297322])


# =======================================================
# LIGHTWEIGHT FOREGROUND PROCESSING & FEATURE MATH
# =======================================================
def extract_foreground_mask(img, kernel_size=9):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    _, img_bw = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    
    labels = label(img_bw_cleaned)
    flat_labels = labels.flat
    flat_bw = img_bw_cleaned.flat
    
    if len(flat_bw) == 0 or np.sum(flat_bw) == 0:
        return np.zeros_like(img_gray)
        
    label_of_largest_region = np.argmax(np.bincount(flat_labels, weights=flat_bw))
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


# =======================================================
# REQUIRED ASSIGNMENT PROTOTYPE FUNCTION (NEURAL NETWORK)
# =======================================================
def detect_olive_fly(image) -> bool:
    """
    Classifies olive flies using an ultra-lightweight Neural Network forward pass.
    Aligns with curriculum matrix guidelines for zero-overhead edge deployment.
    """
    try:
        # 1. Extract and standardize physical attributes
        raw_features = compute_features(image)
        scaled_features = (raw_features - X_MEAN) / X_STD
        
        # 2. Hidden Layer Propagation: Z1 = A1*W1 + B1
        z1 = np.dot(scaled_features, W1) + B1
        a2 = np.maximum(0, z1)  # ReLU non-linear activation function
        
        # 3. Output Layer Propagation: Z2 = A2*W2 + B2
        z2 = np.dot(a2, W2) + B2
        
        # Sigmoid activation function mapping to classification boundaries
        probability = 1 / (1 + np.exp(-np.clip(z2[0], -500, 500)))
        
        return bool(probability >= 0.29)
        
    except Exception:
        return False


# =======================================================
# AUTOMATED EVALUATION TRACKER
# =======================================================
def main():
    args = parser.parse_args()
    
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for filename in glob.glob(str(args.directory)+"/**/*.JPG", recursive=True):
        img = cv2.imread(filename)
        if "not_olive_fly" in filename:
            olive_fly = False
        elif "olive_fly" in filename:
            olive_fly = True
        else:
            print(f"{filename} not labeled.")
            continue

        detection_result = detect_olive_fly(img)

        if olive_fly and detection_result:
            TP += 1
        elif olive_fly and not detection_result:
            FN += 1
        elif not olive_fly and detection_result:
            FP += 1
        else:
            TN += 1
            
        if args.verbose:    
            if detect_olive_fly(img):
                print(f"{filename} contains an olive fly.")
            else:
                print(f"{filename} does not contain an olive fly.")
                
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    if (TP + FP) > 0 and (TP + FN) > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    main()