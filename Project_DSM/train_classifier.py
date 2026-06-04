import os
import glob
import cv2
import numpy as np
from skimage.measure import label

# ----------------------------------------------------
# 1. FOREGROUND EXTRACTION (As provided in your notebook)
# ----------------------------------------------------
def extract_foreground(img, kernel_size=9, background_color=255):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    _, img_bw = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((kernel_size, kernel_size))
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    
    labels = label(img_bw_cleaned)
    flat_labels = labels.flat
    flat_bw = img_bw_cleaned.flat
    
    if len(flat_bw) == 0 or np.sum(flat_bw) == 0:
        return img, np.zeros_like(img_gray)
        
    label_of_largest_region = np.argmax(np.bincount(flat_labels, weights=flat_bw))
    largest_region = (labels == label_of_largest_region)
    
    x, y = np.where(np.invert(largest_region))
    foreground = img.copy()
    foreground[x, y] = background_color
    
    return foreground, largest_region.astype(np.uint8)

# ----------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# ----------------------------------------------------
def extract_advanced_features(img):
    foreground, mask = extract_foreground(img)
    
    # Feature 1: Size (Total pixel area of the insect mask)
    area = float(np.sum(mask))
    
    # Feature 2: Aspect Ratio (Width / Height)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    else:
        aspect_ratio = 1.0
        
    # Feature 3: Color Tone (Mean saturation in HSV space within the mask)
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1]
    mean_saturation = float(cv2.mean(saturation_channel, mask=mask)[0])
    
    return np.array([area, aspect_ratio, mean_saturation])

# ----------------------------------------------------
# 3. CLASS MODEL DEFINITION (From your assignment files)
# ----------------------------------------------------
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=5000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.beta = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _create_matrix_features(self, X):
        return np.column_stack((np.ones(len(X)), X))   

    def fit(self, X, y):
        X_mat = self._create_matrix_features(X)
        n_samples, n_features = X_mat.shape
        self.beta = np.zeros(n_features)
        
        for _ in range(self.num_iterations):
            predictions = self._sigmoid(X_mat.dot(self.beta))
            errors = predictions - y
            gradient = X_mat.T.dot(errors) / n_samples
            self.beta -= self.learning_rate * gradient

# ----------------------------------------------------
# 4. AUTOMATED FOLDER SCANNING AND TRAINING Loop
# ----------------------------------------------------
if __name__ == "__main__":
    # Define directory paths relative to where this script is saved
    neg_dir = "not_olive_fly"
    pos_dir = "olive_fly"
    
    X_list = []
    y_list = []
    
    print(" scanning 'not_olive_fly' folder...")
    # Find all JPG files inside the negative directory (case-insensitive matches included)
    neg_pattern = os.path.join(neg_dir, "*.JPG")
    neg_files = glob.glob(neg_pattern) + glob.glob(os.path.join(neg_dir, "*.jpg"))
    
    for f in neg_files:
        img = cv2.imread(f)
        if img is not None:
            X_list.append(extract_advanced_features(img))
            y_list.append(0)  # Label 0 for everything else
            
    print(f" Loaded {len(y_list)} negative insect images.")
    
    print("\n scanning 'olive_fly' folder...")
    pos_pattern = os.path.join(pos_dir, "*.JPG")
    pos_files = glob.glob(pos_pattern) + glob.glob(os.path.join(pos_dir, "*.jpg"))
    
    current_pos_count = 0
    for f in pos_files:
        img = cv2.imread(f)
        if img is not None:
            X_list.append(extract_advanced_features(img))
            y_list.append(1)  # Label 1 for olive fly
            current_pos_count += 1
            
    print(f" Loaded {current_pos_count} positive olive fly images.")
    
    # Process training matrix if data exists
    if len(X_list) > 0:
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Feature Scaling (Standardization maps data smoothly so Logistic Regression converges)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        # Avoid division-by-zero errors if std is somehow exactly 0
        X_std[X_std == 0] = 1e-8 
        X_scaled = (X - X_mean) / X_std
        
        print("\n Training your resource-efficient model parameters...")
        model = LogisticRegression(learning_rate=0.01, num_iterations=5000)
        model.fit(X_scaled, y)
        
        print("\n=======================================================")
        print("🎉 TRAINING SUCCESSFUL! COPY THESE PARAMETERS FOR YOUR SUBMISSION:")
        print("=======================================================")
        print(f"X_MEAN = np.array({list(X_mean)})")
        print(f"X_STD  = np.array({list(X_std)})")
        print(f"BETA   = np.array({list(model.beta)})")
        print("=======================================================")
    else:
        print("\n❌ Error: No images found. Check that the script is in the same place as your folders.")