import glob
import sys
import argparse
import pathlib
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    prog="OliveFly detection test script",
    description="""
        This scripts tests the olive fly detection algorithm
        against a set of images.""")

parser.add_argument('directory', help='location of the dataset',
                    type=pathlib.Path)
parser.add_argument('--verbose', '-v', action="store_true")


# Trained MLP weights from olive_fly_combined.ipynb
X_MEAN = np.array([3877.458723404255, 1.1174638977603828, 149.3449262126882])
X_STD  = np.array([4801.635565207151, 0.5524351102376486, 51.88207312731143])

W1 = np.array([
    [-1.4787926909184586,  0.9596893415034591,  1.093172432294036,  -1.7454627691021827],
    [-0.26691265456305974, -1.811491088397156,  0.4203204015223413,  0.38288509439103313],
    [-0.6746690403539322,  0.17595712582269765, -1.535109766765984,  1.3510070511987022]
])
B1 = np.array([0.6876098416420647, -0.8068542754680046, 0.7945428630582831, -1.1543043806619804])
W2 = np.array([[-0.814867770461975], [-0.9349578475965044], [-0.8518958607413155], [-1.6780657705355944]])
B2 = np.array([0.24297322103842567])


def extract_foreground_mask(img, kernel_size=9):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    _, img_bw = cv2.threshold(img_gray, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_bw_cleaned = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)

    # Use OpenCV connected components instead of skimage (lighter on Raspberry Pi)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_bw_cleaned, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(img_gray)
    # Label 0 is background; find the largest non-background component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


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
    mean_saturation = float(cv2.mean(hsv[:, :, 1], mask=mask)[0])

    return np.array([area, aspect_ratio, mean_saturation])


def detect_olive_fly(image) -> bool:
    try:
        scaled = (compute_features(image) - X_MEAN) / X_STD
        a2 = np.maximum(0, np.dot(scaled, W1) + B1)  # ReLU hidden layer
        z2 = np.dot(a2, W2) + B2
        probability = 1 / (1 + np.exp(-np.clip(z2[0], -500, 500)))
        return bool(probability >= 0.29)
    except Exception:
        return False


def main():
    args = parser.parse_args()

    TP = TN = FP = FN = 0

    pattern = str(args.directory) + "/**/*.JPG"
    for filename in glob.glob(pattern, recursive=True):
        img = cv2.imread(filename)
        if img is None:
            continue

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
            label = "contains an olive fly." if detection_result else "does not contain an olive fly."
            print(f"{filename} {label}")

    print(f"Summary: True positives {TP}, False positives {FP}")
    print(f"Summary: False negatives {FN}, True negatives {TN}")

    total = TP + TN + FP + FN
    if total > 0:
        print(f"Accuracy: {(TP + TN) / total:.4f}")
    if (TP + FP) > 0 and (TP + FN) > 0:
        print(f"Precision: {TP / (TP + FP):.4f}, Recall: {TP / (TP + FN):.4f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
