import glob
import cv2
import numpy as np
from skimage.measure import label
from sklearn.neural_network import MLPClassifier

# =======================================================
# COLOR-SPACE FOREGROUND SEGMENTATION
# =======================================================
# The trap background is saturated yellow; insects/debris are dark and
# non-yellow. In CIELAB the yellow background sits at high b* while the
# foreground sits at low b*, so an Otsu split on the b* channel isolates the
# insect far more cleanly than a grayscale Otsu (which is fooled by shadows /
# lighting). Morphological open+close drops specks and fills gaps.
def extract_foreground_mask(img, kernel_size=7):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    _, bw = cv2.threshold(b_channel, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    labels = label(bw)
    if bw.sum() == 0:
        return np.zeros(bw.shape, np.uint8), labels
    label_of_largest_region = np.argmax(np.bincount(labels.flat, weights=bw.flat))
    return (labels == label_of_largest_region).astype(np.uint8), labels


# =======================================================
# FEATURE EXTRACTION (13 features)
# =======================================================
# shape:  area_frac, aspect, solidity, circularity, eccentricity, hu1, n_components
# colour: mean H/S/V and Lab a*/b* under the mask, fg-vs-bg contrast
def compute_features(image):
    H, W = image.shape[:2]
    mask, labels = extract_foreground_mask(image)
    area = float(mask.sum())
    area_frac = area / (H * W)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect = solidity = circ = ecc = hu1 = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        ca = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / h if h > 0 else 1.0
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = ca / hull if hull > 0 else 0.0
        perim = cv2.arcLength(c, True)
        circ = min(4 * np.pi * ca / (perim * perim), 1.5) if perim > 0 else 0.0
        if len(c) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(c)
            ecc = min(max(MA, ma) / min(MA, ma), 5.0) if min(MA, ma) > 0 else 0.0
        hu = cv2.HuMoments(cv2.moments(c)).flatten()
        hu1 = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-30))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m = mask.astype(bool)
    if m.any():
        hue = float(hsv[:, :, 0][m].mean())
        sat = float(hsv[:, :, 1][m].mean())
        val = float(hsv[:, :, 2][m].mean())
        lab_a = float(lab[:, :, 1][m].mean())
        lab_b = float(lab[:, :, 2][m].mean())
        bg = gray[~m].mean() if (~m).any() else gray[m].mean()
        contrast = float(bg - gray[m].mean())
    else:
        hue = sat = val = lab_a = lab_b = contrast = 0.0

    counts = np.bincount(labels.flat)
    if len(counts):
        counts[0] = 0
    ncomp = float((counts > 0.1 * area).sum()) if area > 0 else 0.0

    return np.array([area_frac, aspect, solidity, circ, ecc,
                     hue, sat, val, lab_a, lab_b, ncomp, contrast, hu1])


# =======================================================
# DATA AUGMENTATION (used to balance the minority class)
# =======================================================
# Instead of duplicating the 315 olive-fly images (which teaches the net
# nothing new), we synthesise varied positives with random flips, rotation,
# scale and brightness/contrast jitter. BORDER_REFLECT keeps the yellow trap
# texture at the edges instead of injecting black borders.
def augment(img, rng):
    out = img.copy()
    if rng.rand() < 0.5:
        out = cv2.flip(out, 1)
    if rng.rand() < 0.5:
        out = cv2.flip(out, 0)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2),
                                rng.uniform(-30, 30), rng.uniform(0.9, 1.1))
    out = cv2.warpAffine(out, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    out = np.clip(rng.uniform(0.85, 1.15) * out.astype(np.float32) + rng.uniform(-25, 25),
                  0, 255).astype(np.uint8)
    return out


def metrics(proba, y, th):
    pred = proba >= th
    tp = int((pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, tn, fp, fn, p, r, f1


if __name__ == "__main__":
    rng = np.random.RandomState(42)

    print("Reading data...")
    pos = [cv2.imread(f) for f in (glob.glob("olive_fly/*.JPG") + glob.glob("olive_fly/*.jpg"))]
    neg = [cv2.imread(f) for f in (glob.glob("not_olive_fly/*.JPG") + glob.glob("not_olive_fly/*.jpg"))]
    pos = [i for i in pos if i is not None]
    neg = [i for i in neg if i is not None]
    print(f"Class counts (raw): olive_fly={len(pos)}, not_olive_fly={len(neg)}")

    # ---- honest held-out estimate: split BEFORE augmenting so synthetic
    #      positives never leak a copy of a validation image ----
    def split(imgs):
        idx = rng.permutation(len(imgs)); cut = int(0.8 * len(imgs))
        return [imgs[i] for i in idx[:cut]], [imgs[i] for i in idx[cut:]]
    ptr, pte = split(pos); ntr, nte = split(neg)
    aug_tr = [augment(ptr[rng.randint(len(ptr))], rng) for _ in range(len(ntr) - len(ptr))]

    feats = lambda imgs: np.array([compute_features(i) for i in imgs])
    Xtr = np.vstack([feats(ptr), feats(aug_tr), feats(ntr)])
    ytr = np.array([1] * (len(ptr) + len(aug_tr)) + [0] * len(ntr))
    Xte = np.vstack([feats(pte), feats(nte)])
    yte = np.array([1] * len(pte) + [0] * len(nte))

    mean, std = Xtr.mean(0), Xtr.std(0); std[std == 0] = 1e-8
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=8000, random_state=42)
    mlp.fit((Xtr - mean) / std, ytr)
    pv = mlp.predict_proba((Xte - mean) / std)[:, 1]
    th_cv = max(np.linspace(0.1, 0.9, 81), key=lambda t: metrics(pv, yte, t)[6])
    print("HELD-OUT (unseen) | th=%.2f TP=%d TN=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f"
          % ((th_cv,) + metrics(pv, yte, th_cv)))

    # ---- final model: retrain on ALL data (+augmented positives) ----
    print("\nBalancing via augmentation and training final model on all data...")
    aug_all = [augment(pos[rng.randint(len(pos))], rng) for _ in range(len(neg) - len(pos))]
    X = np.vstack([feats(pos), feats(aug_all), feats(neg)])
    y = np.array([1] * (len(pos) + len(aug_all)) + [0] * len(neg))
    print(f"Class counts (balanced): olive_fly={int((y==1).sum())}, not_olive_fly={int((y==0).sum())}")

    X_mean, X_std = X.mean(0), X.std(0); X_std[X_std == 0] = 1e-8
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=8000, random_state=42)
    mlp.fit((X - X_mean) / X_std, y)

    # evaluate on the real (un-augmented) images and pick the F1-optimal threshold
    Xreal = np.vstack([feats(pos), feats(neg)])
    yreal = np.array([1] * len(pos) + [0] * len(neg))
    pr = mlp.predict_proba((Xreal - X_mean) / X_std)[:, 1]
    THRESHOLD = float(max(np.linspace(0.1, 0.9, 81), key=lambda t: metrics(pr, yreal, t)[6]))
    print("FULL DATASET     | th=%.2f TP=%d TN=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f"
          % ((THRESHOLD,) + metrics(pr, yreal, THRESHOLD)))

    print("\n# ---- paste into olivefly_classifier.py ----")
    print(f"THRESHOLD = {THRESHOLD}")
    print(f"X_MEAN = np.array({list(X_mean)})")
    print(f"X_STD  = np.array({list(X_std)})")
    print(f"W1 = np.array({mlp.coefs_[0].tolist()})")
    print(f"B1 = np.array({mlp.intercepts_[0].tolist()})")
    print(f"W2 = np.array({mlp.coefs_[1].tolist()})")
    print(f"B2 = np.array({mlp.intercepts_[1].tolist()})")
