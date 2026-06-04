import numpy as np
import cv2
from skimage.measure import label

# =======================================================
# YOUR MATHEMATICALLY OPTIMIZED TRAINING PARAMETERS
# =======================================================
X_MEAN = np.array([3877.458723404255, 1.1174638977603828, 149.3449262126882])
X_STD  = np.array([4801.635565207151, 0.5524351102376486, 51.88207312731143])
BETA   = np.array([-1.9096005592994554, -0.10140490516913897, -0.08397068506421064, 0.34869182622872813])

# =======================================================
# LIGHTWEIGHT FOREGROUND PROCESSING (Zero training math)
# =======================================================
def extract_foreground_mask(img, kernel_size=9):
    """
    Extracts the binary region of interest for the insect body.
    """
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
    largest_region = (labels == label_of_largest_region)
    return largest_region.astype(np.uint8)

def compute_features(image):
    """
    Computes rapid matrix-based shape and color traits from the image.
    """
    mask = extract_foreground_mask(image)
    
    # Feature 1: Area (Total pixels)
    area = float(np.sum(mask))
    
    # Feature 2: Aspect Ratio (Width / Height)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    else:
        aspect_ratio = 1.0
        
    # Feature 3: Color Saturation Tone inside the mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1]
    mean_saturation = float(cv2.mean(saturation_channel, mask=mask)[0])
    
    return np.array([area, aspect_ratio, mean_saturation])

# =======================================================
# REQUIRED ASSIGNMENT PROTOTYPE FUNCTION
# =======================================================
def detect_olive_fly(image) -> bool:
    """
    Correctly classifies olive flies with the least amount of energy.
    Runs headless and efficiently on a Raspberry Pi 4.
    """
    try:
        # 1. Extract physical attributes from the image matrix
        raw_features = compute_features(image)
        
        # 2. Scale features instantly using dataset parameters to prevent numerical bias
        scaled_features = (raw_features - X_MEAN) / X_STD
        
        # 3. Add an intercept/bias coordinate (Stacking a 1 at the front)
        X_final = np.insert(scaled_features, 0, 1)
        
        # 4. Perform the dot-product matrix multiplication (Logistic Regression)
        z = np.dot(X_final, BETA)
        
        # Sigmoid probability computation (Clipped to protect against math overflow)
        probability = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        # 5. Return True if probability satisfies or exceeds the 50% threshold
        return bool(probability >= 0.5)
        
    except Exception:
        # Defensive programming: fallback choice if image matrix is malformed
        return False