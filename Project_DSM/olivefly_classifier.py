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
# NEURAL NETWORK PARAMETERS (trained by train_classifier.py)
# 13 inputs -> 10 hidden (ReLU) -> 1 output (sigmoid)
# =======================================================
THRESHOLD = 0.69
X_MEAN = np.array([0.08365379220050506, 1.1019440629902508, 0.7894485621677511, 0.4759238328788182, 1.9748598484482682, 36.93152753712406, 124.18664089740895, 91.2644917274316, 130.82020410074938, 143.13262409673217, 2.8606879606879607, 47.52655995600627, 0.5966270939735213])
X_STD  = np.array([0.07383688855191328, 0.5564002947556022, 0.1323079153876091, 0.17416153705423493, 0.7826395771779009, 28.8924255314286, 50.86951592638992, 47.498898650296745, 4.642284170519928, 12.648002022235074, 2.224280248200121, 31.232244507063665, 0.1480814028094457])
W1 = np.array([[-0.6756362089886485, 1.1895597697526856, 0.7951428319275429, -0.2886909636547644, -1.7346553373033191, -2.143105827071289, 0.23042942679204886, 0.7634264425731154, -0.10792898932636773, 1.0852611393812062], [-0.014342194250688802, 0.15281654938532696, 0.016662863615874927, 0.3918204995594099, -0.04708814061635956, 0.0855486102129715, -0.06112879366767353, 0.20267178691501256, -0.05285256188074273, 0.012287432465337079], [-0.04856110403186239, 0.34712891098048676, 0.36966359336558435, -0.49922289247760177, -0.43085016732428094, 0.4550584470709136, -0.12131498254774663, -0.7597344940552506, 0.3168218701141997, -0.3621010068444908], [0.790362236834751, -0.421780305658497, -0.5430206235172743, -0.8751703265165977, 0.451702420591764, 0.27827293779994877, 0.5902952419103288, -0.8293207313071069, -0.3934282571591549, -0.9320734617102765], [0.25962226930809246, 0.4655680717620421, 0.29328394767117044, 0.5671254514211075, -0.28776156691470284, 0.18781434637338235, -0.038910928745512864, -0.5703123401408775, 0.12281143905720232, -0.26233609262825974], [0.9835574773635544, 0.19397052940365117, 0.21796562785241724, 0.2018324663337463, -0.3003446207377556, 0.42057225722751135, -1.2137943076187765, -1.114572986912628, -0.17826639729590424, -0.017878263611014078], [-0.10904889852744672, 0.034698076661597965, 1.7785811765855966, -0.1848677866804208, -0.4078999384175894, 0.2339651937590806, -1.533543088981983, 1.1006401916750657, -0.33671591222048214, 0.12384676826091476], [-0.6344764565090913, 0.14715668481205113, -0.29005104337104903, 0.4383460254529539, -0.6418270222741085, 0.44875711789616507, 0.6953837010496497, -0.26207838407891537, -0.5482446013397709, 0.14284545392212442], [0.8059153490936479, -0.5296297130548354, -0.3906904566670874, 0.2220492159185696, 0.20743371540895325, 0.226176639020651, -0.12438542730186171, 0.2743255576690131, -0.18268717433739134, 0.08169815548855164], [-0.7293408046586695, 0.06400309636082663, -0.814147626606113, -0.44419241075246413, 0.3779959225342873, -0.6196826689722652, 0.3878896979285695, -1.2036367451405898, -1.2362019170954648, -0.6520916825529119], [-0.4075717933582585, -0.13884048202367086, 0.12983688327183543, 0.0467224162004279, 0.4545569646791144, -0.10243735550216646, -0.029654386964009804, -0.2297451337605024, -0.08664859721436352, -0.561559337884911], [-0.68545976938811, -0.9328703168392349, 0.9360511059257195, 0.020657955443650237, -0.7995608321476106, -0.392722619775219, -0.23292905413320933, -0.09241098506605205, 0.9654159834077508, 0.13494113291782386], [0.03373009486483809, 0.8643963203514412, 0.28104620107797096, -0.7182027504632529, -1.1734117793390575, 0.22446744767064544, -0.7224181473901377, 0.41740960436778496, 0.7860651405289539, 0.5239814014673649]])
B1 = np.array([0.566685811642649, 0.8587219536226088, -1.603666751961894, -0.5098417813487097, 0.07430696048825487, -0.8801548031167337, 0.9870546083680783, 0.08177111438161713, -0.6640332736300643, 0.6510148332038598])
W2 = np.array([[1.5003926132815617], [-1.2849527351274128], [2.361170844948447], [-0.7519866593339499], [-1.5698493113469314], [-2.5009436400871268], [1.618950697349959], [1.0179496338450933], [-1.6107857420306018], [-0.7281050879075026]])
B2 = np.array([0.845348646913506])


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
# FEATURE EXTRACTION (13 features) — must match train_classifier.py
# =======================================================
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

        return bool(probability >= THRESHOLD)

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
            if detection_result:
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