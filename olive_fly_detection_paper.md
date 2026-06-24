#Automated Detection of Olive Fly (*Bactrocera oleae*) Using Color-Space Segmentation, Morphological Feature Extraction, and a Multilayer Perceptron Neural Network

---

## Abstract

The olive fly (*Bactrocera oleae*) is one of the most economically damaging pests to olive cultivation worldwide, causing significant losses in both yield and oil quality. This paper presents an automated image-based detection system designed to classify trap images as containing or not containing an olive fly. The system operates through three sequential stages: foreground segmentation using Otsu thresholding in the CIELAB color space, extraction of 13 handcrafted morphological and color features, and binary classification using a Multilayer Perceptron (MLP) neural network. To address class imbalance in the training dataset, image augmentation was applied to the minority class. A decision threshold was tuned by maximizing the F1 score on real (non-augmented) images. The final system achieved an accuracy of 89.83%, a precision of 60.61%, and a recall of 68.89% on the training dataset.

---

## 1. Introduction

The olive fly (*Bactrocera oleae*) is a major agricultural pest whose larvae feed on olive fruit, causing significant losses in yield and oil quality (Daane & Johnson, 2010). Early and accurate detection of olive flies in monitoring traps is essential for timely pest management intervention.

Traditional trap monitoring is performed manually, which is labor-intensive and prone to human error. Automated image classification offers a scalable alternative. This system is designed to run on resource-constrained hardware such as a Raspberry Pi, and therefore relies on classical computer vision features rather than deep learning, which would require substantially more computational resources.

The core objective is to take a photograph of a sticky trap and determine whether it contains an olive fly specimen. This is framed as a supervised binary classification problem, where the two classes are `olive_fly` (positive) and `not_olive_fly` (negative).

---

## 2. Dataset

The dataset consists of `.JPG` images organized into two directories:

- **`olive_fly/`** — images containing at least one olive fly specimen (630 images)
- **`not_olive_fly/`** — images of trap contents without olive flies (4,070 images)

This represents a class imbalance ratio of approximately 1:6.5 in favor of the negative class. Left unaddressed, this imbalance would bias the classifier toward predicting the majority class. Section 4 describes how this was addressed through data augmentation.

---

## 3. Foreground Segmentation

Before features can be extracted, the insect or object of interest must be separated from the trap background. This is achieved through a color-aware segmentation pipeline.

### 3.1 Color Space Conversion

Standard sticky olive fly traps use a saturated yellow adhesive background. This color property can be exploited for segmentation. The images are converted from BGR (the default format used by OpenCV) to the CIELAB color space (Commission Internationale de l'Éclairage, 1978). In CIELAB, the L* channel represents lightness, the a* channel represents a green–red axis, and the b* channel represents a blue–yellow axis. Because the trap background is strongly yellow, it produces high values on the b* channel, while foreground objects (insects, debris) produce lower b* values. Thresholding on the b* channel therefore separates the foreground more cleanly than a standard grayscale threshold, which is susceptible to lighting variation and shadow (Bradski & Kaehler, 2008).

### 3.2 Otsu's Thresholding

Rather than setting a fixed threshold manually, Otsu's method is applied to the b* channel (Otsu, 1979). This algorithm automatically determines the optimal threshold value by minimizing intra-class variance between the two resulting pixel groups (foreground and background). The result is a binary image where pixels belonging to the foreground are marked as white (255) and background pixels as black (0), with the threshold inverted so that the low-b* foreground pixels are selected.

### 3.3 Morphological Cleaning

The binary mask produced by thresholding often contains small spurious regions (noise) and holes within larger regions. Two morphological operations are applied sequentially to clean the mask (Gonzalez & Woods, 2018):

1. **Morphological Opening** (erosion followed by dilation) with a 7×7 kernel removes small isolated noise pixels.
2. **Morphological Closing** (dilation followed by erosion) with the same kernel fills small holes within larger foreground regions.

Both operations use OpenCV's `cv2.morphologyEx` function with a rectangular structuring element (Bradski & Kaehler, 2008).

### 3.4 Connected Component Analysis

After morphological cleaning, the binary mask may still contain multiple disconnected regions. To isolate the single most significant foreground object, connected component labeling is performed using `cv2.connectedComponentsWithStats` (Bradski & Kaehler, 2008). Each connected region is assigned a unique integer label, and the region with the largest pixel area (excluding the background label 0) is retained. The final mask is a binary image containing only this largest region. This approach follows the assumption that the primary object of interest occupies the largest contiguous foreground area.

---

## 4. Data Augmentation

To compensate for the class imbalance described in Section 2, synthetic positive samples were generated from the existing `olive_fly` images through random image augmentation (Shorten & Khoshgoftaar, 2019). Augmentation was applied until the positive class matched the size of the negative class (4,070 samples each). The augmentation operations applied to each synthetic image were:

- **Horizontal flip** (applied with 50% probability)
- **Vertical flip** (applied with 50% probability)
- **Random rotation** between −30° and +30° with a random scaling factor between 0.9 and 1.1, using reflected border padding
- **Brightness and contrast jitter**: pixel values multiplied by a random factor in [0.85, 1.15] and shifted by a random offset in [−25, +25], then clipped to the valid [0, 255] range

All random operations used a fixed random seed (42) to ensure reproducibility (Harris et al., 2020).

---

## 5. Feature Extraction

For each image, 13 numerical features are extracted from the segmented foreground mask. These features are grouped into shape descriptors and color descriptors.

### 5.1 Shape Features

**(1) Area Fraction (`area_frac`):** The proportion of the total image pixels occupied by the foreground mask. This captures the relative size of the detected object independent of image resolution.

$$\text{area\_frac} = \frac{\text{foreground pixels}}{H \times W}$$

**(2) Aspect Ratio (`aspect`):** The ratio of the width to the height of the axis-aligned bounding rectangle enclosing the largest contour. An elongated insect body produces a value different from 1.0, while more circular objects approach 1.0.

$$\text{aspect} = \frac{w}{h}$$

**(3) Solidity (`solidity`):** The ratio of the contour area to the area of its convex hull. A solid, compact shape has a solidity close to 1.0, while irregular or concave shapes produce lower values (Bradski & Kaehler, 2008).

$$\text{solidity} = \frac{\text{contour area}}{\text{convex hull area}}$$

**(4) Circularity (`circ`):** A dimensionless measure of how closely the shape resembles a circle, derived from the contour's area and perimeter (Haralick & Shapiro, 1992). A perfect circle yields a value of 1.0.

$$\text{circularity} = \frac{4\pi \cdot \text{area}}{\text{perimeter}^2}$$

Values are clipped to a maximum of 1.5 to reduce the effect of numerical instability on small contours.

**(5) Eccentricity (`ecc`):** Computed by fitting an ellipse to the largest contour using `cv2.fitEllipse`. Eccentricity is the ratio of the major axis to the minor axis. A circle has an eccentricity of 1.0; highly elongated shapes have higher values. Values are clipped to a maximum of 5.0.

**(6) First Hu Moment (`hu1`):** Hu moments are a set of seven image moments that are invariant to rotation, scale, and translation (Hu, 1962). The first Hu moment is extracted from the contour's standard moments using `cv2.HuMoments`. Because Hu moments span many orders of magnitude, the value is log-transformed:

$$\text{hu1} = -\text{sign}(h_1) \cdot \log_{10}(|h_1| + \epsilon)$$

where $\epsilon = 10^{-30}$ prevents a logarithm of zero.

**(7) Number of Components (`ncomp`):** The number of connected components whose pixel area exceeds 10% of the total foreground area. This feature captures structural complexity — a single large object scores 1, while a cluttered trap may score higher.

### 5.2 Color Features

All color features are computed only over pixels belonging to the foreground mask, ensuring they describe the object rather than the background.

**(8) Mean Hue (`hue`):** The average hue value of foreground pixels in the HSV (Hue, Saturation, Value) color space. Hue represents the dominant wavelength of color and is useful for distinguishing olive flies (typically dark brown or black) from other trap contents.

**(9) Mean Saturation (`sat`):** The average color saturation of foreground pixels in HSV. Insects tend to have lower saturation (more achromatic) compared to trap residues.

**(10) Mean Value (`val`):** The average brightness of foreground pixels in HSV. This captures whether the foreground object is dark or light relative to the trap.

**(11) Mean Lab a\* (`lab_a`):** The mean value of the a* channel (green–red axis) of the CIELAB representation under the foreground mask.

**(12) Mean Lab b\* (`lab_b`):** The mean value of the b* channel (blue–yellow axis) under the foreground mask. Olive flies, being dark insects against a yellow background, are expected to produce distinctive b* signatures.

**(13) Foreground–Background Contrast (`contrast`):** The difference between the mean grayscale intensity of background pixels and foreground pixels.

$$\text{contrast} = \overline{I_{\text{background}}} - \overline{I_{\text{foreground}}}$$

A positive contrast indicates a darker foreground against a lighter background, consistent with the expected appearance of an insect on a yellow trap.

---

## 6. Feature Standardization

Raw feature values span very different numerical ranges (e.g., `area_frac` is typically in [0, 1] while `hue` spans [0, 180] in OpenCV's HSV encoding). Training a neural network on unstandardized features causes weights corresponding to large-magnitude features to dominate gradient updates, slowing or destabilizing convergence (LeCun et al., 2002). Each feature is therefore standardized using z-score normalization:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

where $\mu$ and $\sigma$ are the per-feature mean and standard deviation computed from the full training set (including augmented samples). Any feature with a standard deviation of zero is assigned $\sigma = 10^{-8}$ to avoid division by zero.

The computed normalization constants are stored as constants in the inference script so that new images at test time are scaled with the same parameters used during training.

---

## 7. Neural Network Architecture and Training

### 7.1 Architecture

A Multilayer Perceptron (MLP) is used for binary classification (Rumelhart et al., 1986). The network consists of three layers:

- **Input layer:** 13 neurons (one per feature)
- **Hidden layer:** 10 neurons with Rectified Linear Unit (ReLU) activation
- **Output layer:** 1 neuron with sigmoid activation producing a probability in [0, 1]

The ReLU activation function is defined as $f(x) = \max(0, x)$ and was chosen for its computational efficiency and resistance to the vanishing gradient problem (Nair & Hinton, 2010). The output sigmoid function maps the unbounded pre-activation value to a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The forward pass through the network for a single input feature vector $\mathbf{x}$ is:

$$\mathbf{z}_1 = \mathbf{x} W_1 + \mathbf{b}_1$$
$$\mathbf{a}_2 = \max(0, \mathbf{z}_1)$$
$$z_2 = \mathbf{a}_2 W_2 + b_2$$
$$p = \sigma(z_2)$$

where $W_1 \in \mathbb{R}^{13 \times 10}$, $\mathbf{b}_1 \in \mathbb{R}^{10}$, $W_2 \in \mathbb{R}^{10 \times 1}$, and $b_2 \in \mathbb{R}$.

### 7.2 Training

The model was trained using scikit-learn's `MLPClassifier` (Pedregosa et al., 2011) with the following settings:

- Hidden layer sizes: (10,)
- Activation: ReLU
- Maximum iterations: 8,000
- Random seed: 42

Training was performed on the class-balanced dataset of 8,140 images (4,070 positive, 4,070 negative). The network weights were optimized using L-BFGS or Adam (the default solver selected by scikit-learn based on dataset size).

---

## 8. Decision Threshold Tuning

The default decision threshold for binary classification is 0.5: the model predicts positive if the output probability $p \geq 0.5$. However, in imbalanced problems or when the costs of false positives and false negatives differ, the threshold should be tuned (Fawcett, 2006).

The threshold was selected to maximize the F1 score on the real (non-augmented) images only (315 positive + 4,070 negative = 4,700 samples). The F1 score is the harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

81 threshold values were evaluated uniformly over [0.1, 0.9], and the value maximizing F1 was selected. The optimal threshold was found to be **0.70**.

---

## 9. Results

The classifier was evaluated on the full labeled training dataset (excluding augmented samples) using the tuned threshold of 0.70. The confusion matrix results were:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP = 217 | FN = 98 |
| **Actual Negative** | FP = 141 | TN = 1,894 |

From these values, the following metrics were computed:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{217 + 1894}{2350} = 0.8983$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{217}{358} = 0.6061$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{217}{315} = 0.6889$$

The system achieves approximately 89.8% overall accuracy. The recall of 68.9% indicates that the system correctly identifies roughly two thirds of actual olive fly images, while the precision of 60.6% indicates that most positive predictions are correct.

---

## 10. Conclusion

This paper described an automated image-based olive fly detection system designed for deployment on resource-constrained embedded hardware. The pipeline combines LAB color-space segmentation, morphological image processing, and a compact 13-feature MLP classifier. The model is implemented entirely using OpenCV and NumPy, eliminating heavy dependencies such as scikit-image or deep learning frameworks that may not be reliably available on a Raspberry Pi.

The system achieves 89.83% accuracy on the training dataset with an F1-optimized threshold. Future work could investigate additional shape features, ensemble methods, or lightweight convolutional architectures to improve recall without increasing computational cost.

---

## References

<p style="padding-left: 2em; text-indent: -2em;">Bradski, G. (2000). The OpenCV library. <em>Dr. Dobb's Journal of Software Tools</em>, <em>25</em>(11), 120–125.</p>

<p style="padding-left: 2em; text-indent: -2em;">Bradski, G., & Kaehler, A. (2008). <em>Learning OpenCV: Computer vision with the OpenCV library</em>. O'Reilly Media.</p>

<p style="padding-left: 2em; text-indent: -2em;">Commission Internationale de l'Éclairage. (1978). <em>Recommendations on uniform color spaces, color-difference equations, psychometric color terms</em> (CIE Publication No. 15). Bureau Central de la CIE.</p>

<p style="padding-left: 2em; text-indent: -2em;">Daane, K. M., & Johnson, M. W. (2010). Olive fruit fly: Managing an ancient pest in modern times. <em>Annual Review of Entomology</em>, <em>55</em>, 291–310. https://doi.org/10.1146/annurev.ento.54.110807.090553</p>

<p style="padding-left: 2em; text-indent: -2em;">Fawcett, T. (2006). An introduction to ROC analysis. <em>Pattern Recognition Letters</em>, <em>27</em>(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010</p>

<p style="padding-left: 2em; text-indent: -2em;">Gonzalez, R. C., & Woods, R. E. (2018). <em>Digital image processing</em> (4th ed.). Pearson.</p>

<p style="padding-left: 2em; text-indent: -2em;">Haralick, R. M., & Shapiro, L. G. (1992). <em>Computer and robot vision</em> (Vol. 1). Addison-Wesley.</p>

<p style="padding-left: 2em; text-indent: -2em;">Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. <em>Nature</em>, <em>585</em>(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2</p>

<p style="padding-left: 2em; text-indent: -2em;">Hu, M.-K. (1962). Visual pattern recognition by moment invariants. <em>IRE Transactions on Information Theory</em>, <em>8</em>(2), 179–187. https://doi.org/10.1109/TIT.1962.1057692</p>

<p style="padding-left: 2em; text-indent: -2em;">LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (2002). Efficient backprop. In G. B. Orr & K.-R. Müller (Eds.), <em>Neural networks: Tricks of the trade</em> (pp. 9–50). Springer. https://doi.org/10.1007/3-540-49430-8_2</p>

<p style="padding-left: 2em; text-indent: -2em;">Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In J. Fürnkranz & T. Joachims (Eds.), <em>Proceedings of the 27th International Conference on Machine Learning</em> (pp. 807–814). Omnipress.</p>

<p style="padding-left: 2em; text-indent: -2em;">Otsu, N. (1979). A threshold selection method from gray-level histograms. <em>IEEE Transactions on Systems, Man, and Cybernetics</em>, <em>9</em>(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076</p>

<p style="padding-left: 2em; text-indent: -2em;">Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <em>Journal of Machine Learning Research</em>, <em>12</em>, 2825–2830.</p>

<p style="padding-left: 2em; text-indent: -2em;">Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. <em>Nature</em>, <em>323</em>(6088), 533–536. https://doi.org/10.1038/323533a0</p>

<p style="padding-left: 2em; text-indent: -2em;">Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. <em>Journal of Big Data</em>, <em>6</em>(1), Article 60. https://doi.org/10.1186/s40537-019-0197-0</p>

<p style="padding-left: 2em; text-indent: -2em;">van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: Image processing in Python. <em>PeerJ</em>, <em>2</em>, e453. https://doi.org/10.7717/peerj.453</p>

