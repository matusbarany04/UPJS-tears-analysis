#!/usr/bin/env python3
import argparse
import sys
import scipy
from seaborn._marks import area
from sympy.sets import conditionset

CLASSES = [
    "Diabetes",
    "Skleroza",
    "Glaukom",
    "Suche oko",
    "Zdravi ludia"
]

import numpy as np
import cv2
import os
from skimage.morphology import skeletonize
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
import math
from skimage.measure import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk
import warnings
from skimage.measure import label, regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os


def box_counting_fractal_dimension(image_array):
    """
    Calculates the fractal dimension of a 2D image using the box-counting method.

    Parameters:
    - image_array: 2D NumPy array (can be binary or grayscale, will be binarized > 0)

    Returns:
    - fd: The fractal dimension (float), typically between 1.0 and 2.0.
    """
    # 1. Ensure the image is strictly binary (background=0, crystal=1)
    Z = (image_array > 0)

    # Extract the Y, X coordinates of all crystal pixels
    pixels = np.argwhere(Z)
    if len(pixels) == 0:
        return 0.0  # Return 0 if the image is completely empty

    Ly, Lx = Z.shape

    # 2. Define the varying box sizes
    # We create a logarithmic scale of box sizes from 1 pixel up to 1/5th of the image size
    max_box_size = min(Lx, Ly) // 5
    # Generate logarithmically spaced sizes and ensure they are unique integers
    sizes = np.logspace(0.1, np.log10(max_box_size), num=20, base=10)
    sizes = np.unique(np.floor(sizes)).astype(int)

    counts = []

    # 3. Count the boxes for each grid size
    for size in sizes:
        # Determine the number of boxes in X and Y directions
        Nx = int(np.ceil(Lx / size))
        Ny = int(np.ceil(Ly / size))

        # Create the grid boundaries (bins)
        xbins = np.arange(0, Nx * size + 1, size)
        ybins = np.arange(0, Ny * size + 1, size)

        # Drop the pixels into the grid using a 2D histogram
        H, _, _ = np.histogram2d(pixels[:, 0], pixels[:, 1], bins=(ybins, xbins))

        # Count how many boxes contain at least 1 pixel (H > 0)
        counts.append(np.sum(H > 0))

    # 4. Calculate the slope of the line of best fit (Fractal Dimension)
    # x-axis: log(1 / box_size)
    # y-axis: log(number of filled boxes)
    x_vals = np.log(1.0 / sizes)
    y_vals = np.log(counts)

    # np.polyfit returns [slope, intercept]
    coeffs = np.polyfit(x_vals, y_vals, 1)
    fractal_dimension = coeffs[0]

    return fractal_dimension


def calculate_skeleton_entropies(skeleton_array):
    """
    Calculates both Global and Local entropy for a binary skeleton image.
    """
    # 1. Global Binary Entropy (Effectively measures Skeleton Density)
    global_ent = shannon_entropy(skeleton_array)

    # 2. Local Spatial Entropy (Measures structural chaos)
    # Convert skeleton to an 8-bit integer array as required by skimage rank filters
    skel_8bit = (skeleton_array > 0).astype(np.uint8) * 255

    # Define the neighborhood size (a circle with a radius of 10 pixels)
    neighborhood = disk(10)

    # Calculate entropy for every local neighborhood
    # Catching warnings here because pure black regions might throw a low-contrast warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_ent_map = entropy(skel_8bit, neighborhood)

    # The final feature is the average chaos across the entire image
    mean_local_ent = np.mean(local_ent_map)

    return global_ent, mean_local_ent


def fft_metric(cropped_image):
    line_thickness = 20

    f_transform = np.fft.fft2(cropped_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    log_magnitude = np.log(np.abs(f_transform_shifted) + 1e-8)

    height, width = cropped_image.shape
    cy, cx = height // 2, width // 2
    mask_radius = 100
    percentile = 70

    y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
    distance_from_center = np.sqrt(x ** 2 + y ** 2)
    min_val = np.min(log_magnitude)
    log_magnitude[distance_from_center <= mask_radius] = min_val

    threshold_value = np.percentile(log_magnitude, percentile)
    binary_fft = (log_magnitude >= threshold_value).astype(int)

    cross_mask = np.zeros_like(binary_fft)
    cross_mask[cy - line_thickness: cy + line_thickness + 1, :] = 1
    cross_mask[:, cx - line_thickness: cx + line_thickness + 1] = 1

    pixels_in_cross = np.sum(binary_fft * cross_mask)
    total_white_pixels = np.sum(binary_fft)

    return pixels_in_cross / (total_white_pixels + 1e-10)


def count_components(skeleton, min_size=5, max_size=math.inf):
    labeled = label(skeleton, connectivity=2)

    count = 0
    for r in regionprops(labeled):
        if min_size < r.area < max_size:
            count += 1

    return count


def get_spot_count(img):
    SPOT_SIZE = 3

    def get_spot_contrast(gray, binary, x, y, spot_size, n):
        sum_spot = 0.0
        sum_out = 0.0

        count_spot = 0
        count_out = 0

        h, w = gray.shape

        for i in range(x - n - spot_size, x + n + spot_size + 1):
            for j in range(y - n - spot_size, y + n + spot_size + 1):

                if not (0 <= i < w and 0 <= j < h):
                    continue

                val = float(gray[j, i])  # 👈 avoid overflow

                # inside square spot
                if binary[j][i]:
                    sum_spot += val
                    count_spot += 1
                else:
                    sum_out += val
                    count_out += 1

        if count_spot == 0 or count_out == 0:
            return 0

        mean_spot = sum_spot / count_spot
        mean_out = sum_out / count_out

        return mean_spot / mean_out

    gray = img

    kernel = np.ones((25, 25), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.GaussianBlur(tophat, (5, 5), 0)

    n = 100
    flat = gray.flatten()
    idx = np.argsort(flat)[:n]  # indices of n darkest pixels
    values = flat[idx]
    avg_n_min = np.mean(values)
    print(avg_n_min)
    _, binary = cv2.threshold(tophat, int(110) - avg_n_min, 255, cv2.THRESH_BINARY)

    kernel_small = np.ones((SPOT_SIZE, SPOT_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    spots = []

    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]

        if area > 0 and area < 10:  # 👈 key condition
            x, y = centroids[j]

            if gray[int(y), int(x)] > 180 and get_spot_contrast(tophat, binary, int(x), int(y), SPOT_SIZE, 2) > 1.8:
                spots.append((int(x), int(y), area))

    #  coords = np.array([(x, y) for x, y, area in spots])

    # spread = np.mean(np.var(coords, axis=0))
    # print("Spread (variance):", spread)

    output = img.copy()
    for x, y, area in spots:
        cv2.circle(output, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(output, f"{area}", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                    1)  # ------------------------- # Step 8: show # -------------------------

    # cv2.imshow("Top-hat", tophat)
    # cv2.imshow("Binary", binary)
    # cv2.imshow("Bright spots", output)
    # cv2.waitKey(0)

    return len(spots)


# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class TearCrystalNet(nn.Module):
    # Added num_classes parameter so we can use this for both Stage 1 and Stage 2!
    def __init__(self, num_features, num_classes):
        super(TearCrystalNet, self).__init__()

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.cnn_fc = nn.Linear(2048, 64)

        self.mlp_branch = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fusion_layers = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)  # Dynamically outputs 1 (Binary) or 4 (Multi-class)
        )

    def forward(self, img_stack, feature_vector):
        x_img = self.cnn_branch(img_stack)
        x_img = F.relu(self.cnn_fc(x_img))

        x_feat = self.mlp_branch(feature_vector)

        combined = torch.cat((x_img, x_feat), dim=1)
        return self.fusion_layers(combined)


# ==========================================
# 2. THE LIVE PREDICTOR CLASS
# ==========================================
class TearPredictor:
    def __init__(self, binary_weights_path, disease_weights_path, scaler_stage1_path, scaler_stage2_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Scaler
        self.scaler_stage1 = joblib.load(scaler_stage1_path)
        self.scaler_stage2 = joblib.load(scaler_stage2_path)

        # 2. Load Stage 1 Model (Binary: 1 output)
        self.stage1_model = TearCrystalNet(num_features=7, num_classes=1).to(self.device)
        self.stage1_model.load_state_dict(torch.load(binary_weights_path, map_location=self.device))
        self.stage1_model.eval()  # Lock for inference

        # 3. Load Stage 2 Model (Disease: 4 outputs)
        self.stage2_model = TearCrystalNet(num_features=2, num_classes=4).to(self.device)
        self.stage2_model.load_state_dict(torch.load(disease_weights_path, map_location=self.device))
        self.stage2_model.eval()  # Lock for inference

        self.disease_names = ['Diabetes', 'Sklerosis', 'Glaucoma', 'Dry Eye']

    def get_data(self, image_path):
        def load_image(image_path):
            # load is np gray scale array
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img

        def change_contrast(image, low=10, high=98):
            p_low, p_high = np.percentile(image, (low, high))
            return rescale_intensity(image, in_range=(p_low, p_high))

        def process_skeleton(img):
            img = change_contrast(img)
            th = threshold_otsu(img)
            binary = img > th

            skell = skeletonize(binary)

            return binary, skell

        def crop_image(image_to_crop, top=10, bottom=575 - 45, left=94, right=704 - 90):
            return image_to_crop[top:bottom, left:right]

        img = normalize(crop_image(load_image(image_path)),image_path)

        processed, skell = process_skeleton(img)

        return img, processed, skell

    def extract_features(self, img_crop, img_bin, img_skel, stage1):
        """Runs your math functions and scales the output."""
        fft_val = fft_metric(img_crop)
        fd_bin = box_counting_fractal_dimension(img_bin)
        fd_skel = box_counting_fractal_dimension(img_skel)
        glob_ent_bin, loc_ent_bin = calculate_skeleton_entropies(img_bin)
        glob_ent_skel, loc_ent_skel = calculate_skeleton_entropies(img_skel)
        n_comps = count_components(img_skel)
        n_spots = get_spot_count(img_crop)

        # Must match the exact order of `feature_cols` from training!
        if stage1:
            raw_features = np.array([[
                fft_val, fd_bin, fd_skel, glob_ent_bin, loc_ent_bin,
                glob_ent_skel, loc_ent_skel
            ]])
        else:
            raw_features = np.array([[
                n_comps, n_spots
            ]])

        # Scale the features
        if stage1:
            scaled_features = self.scaler_stage1.transform(raw_features)
        else:
            scaled_features = self.scaler_stage2.transform(raw_features)
        return torch.tensor(scaled_features, dtype=torch.float32).to(self.device)

    def predict(self, image_path):
        """The main pipeline: Image -> Stage 1 -> (if sick) -> Stage 2 -> Diagnosis"""
        # 1. Load and prepare data

        img_crop, img_bin, img_skel = self.get_data(image_path)
        features_tensor_stage1 = self.extract_features(img_crop, img_bin, img_skel, stage1=True)
        features_tensor_stage2 = self.extract_features(img_crop, img_bin, img_skel, stage1=False)

        # Stack images for CNN: shape (1, 2, H, W)
        img_stack = np.stack([img_bin, img_skel], axis=0).astype(np.float32)
        img_tensor = torch.from_numpy(img_stack).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # ==============================
            # STAGE 1: Is the patient sick?
            # ==============================
            binary_output = self.stage1_model(img_tensor, features_tensor_stage1).squeeze()
            is_sick_prob = torch.sigmoid(binary_output).item()

            # Using 0.5 as threshold, but you can adjust this if you want to be more sensitive
            if is_sick_prob < 0.65:
                print("Patient is healthy, he can go home play videogames!")
                return "Healthy", (1.0 - is_sick_prob)
                return
            else:
                print(
                    f"Patient is likely sick (Confidence: {is_sick_prob * 100:.1f}%), running Stage 2 for detailed diagnosis...")

            # ==============================
            # STAGE 2: Which disease is it?
            # ==============================
            disease_output = self.stage2_model(img_tensor, features_tensor_stage2)
            disease_probs = torch.softmax(disease_output, dim=1).squeeze()

            max_prob, predicted_class_idx = torch.max(disease_probs, 0)
            diagnosis = self.disease_names[predicted_class_idx.item()]


            return diagnosis, max_prob.item() * 100

def branching_factor(skeleton):
    skel = skeleton.astype(np.uint8)

    # 8-neighborhood kernel (center excluded)
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # count neighbors for each pixel
    neighbor_count = scipy.ndimage.convolve(skel, kernel, mode='constant', cval=0)

    # only consider skeleton pixels
    degrees = neighbor_count * skel

    endpoints = np.sum(degrees == 1)
    junctions = np.sum(degrees >= 3)

    bf = (junctions / endpoints) if endpoints > 0 else 0.0

    return bf


def crop_image(image_to_crop, top=10, bottom=575 - 45, left=94, right=704 - 90):
    return image_to_crop[top:bottom, left:right]


def change_contrast(image, low=10, high=98):
    p_low, p_high = np.percentile(image, (low, high))
    return rescale_intensity(image, in_range=(p_low, p_high))


def process_skeleton(img):
    img = change_contrast(img)
    th = threshold_otsu(img)
    binary = img > th

    skell = skeletonize(binary)

    return binary, skell


def count_components(skeleton, min_size=5, max_size=math.inf):
    labeled = label(skeleton, connectivity=2)

    count = 0
    for r in regionprops(labeled):
        if min_size < r.area < max_size:
            count += 1

    return count


def load_image(image_path):
    # load is np gray scale array
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

def normalize(img, image_path, target=50000.0):

    def load_real_size(path) -> float:
        with open(path.replace("_1.bmp", ""), "r", encoding="latin1", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if line.startswith(r"\Scan Size:"):
                    # example: \Scan Size: 92516.8 nm
                    parts = line.split(":")[1].strip().split()
                    return float(parts[0])  # value in nm

        assert False

    area = load_real_size(image_path)

    ratio = target / area  # >1 enlarge, <1 shrink
    h, w = img.shape[:2]

    # # -------------------------
    # # 1. SCALE UP (small images)
    # # -------------------------
    # # 385 is aproximately  50000/92500
    if area < 51000:
        new_w = int(281.0)
        new_h = int(281.0)

        resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        return resized
    # # -------------------------
    # 2. CROP (large images)
    # -------------------------
    else:
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # center crop
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2

        cropped_center = img[
            start_y:start_y + new_h,
            start_x:start_x + new_w
        ]

        return cropped_center


def save_image(img: np.ndarray, filename: str, out_dir="./"):
    if img is None:
        raise ValueError("Image is None")

    # convert bool -> uint8
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    # normalize other weird types
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)

    return path


def get_spot_count(img):
    SPOT_SIZE = 3

    def get_spot_contrast(gray, binary, x, y, spot_size, n):
        sum_spot = 0.0
        sum_out = 0.0

        count_spot = 0
        count_out = 0

        h, w = gray.shape

        for i in range(x - n - spot_size, x + n + spot_size + 1):
            for j in range(y - n - spot_size, y + n + spot_size + 1):

                if not (0 <= i < w and 0 <= j < h):
                    continue

                val = float(gray[j, i])

                # inside square spot
                if binary[j][i]:
                    sum_spot += val
                    count_spot += 1
                else:
                    sum_out += val
                    count_out += 1

        if count_spot == 0 or count_out == 0:
            return 0

        mean_spot = sum_spot / count_spot
        mean_out = sum_out / count_out

        return mean_spot / mean_out

    gray = img

    kernel = np.ones((25, 25), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.GaussianBlur(tophat, (5, 5), 0)

    n = 100
    flat = gray.flatten()
    idx = np.argsort(flat)[:n]  # indices of n darkest pixels
    values = flat[idx]
    avg_n_min = np.mean(values)
    _, binary = cv2.threshold(tophat, int(110) - avg_n_min, 255, cv2.THRESH_BINARY)

    kernel_small = np.ones((SPOT_SIZE, SPOT_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    spots = []

    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]

        if area > 0 and area < 10:
            x, y = centroids[j]

            if gray[int(y), int(x)] > 180 and get_spot_contrast(tophat, binary, int(x), int(y), SPOT_SIZE, 2) > 1.8:
                spots.append((int(x), int(y), area))

    output = img.copy()
    for x, y, area in spots:
        cv2.circle(output, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(output, f"{area}", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0),
                    1)  # ------------------------- # Step 8: show # -------------------------

    return len(spots)


def box_counting_fractal_dimension(image_array):
    """
    Calculates the fractal dimension of a 2D image using the box-counting method.

    Parameters:
    - image_array: 2D NumPy array (can be binary or grayscale, will be binarized > 0)

    Returns:
    - fd: The fractal dimension (float), typically between 1.0 and 2.0.
    """
    # 1. Ensure the image is strictly binary (background=0, crystal=1)
    Z = (image_array > 0)

    # Extract the Y, X coordinates of all crystal pixels
    pixels = np.argwhere(Z)
    if len(pixels) == 0:
        return 0.0  # Return 0 if the image is completely empty

    Ly, Lx = Z.shape

    # 2. Define the varying box sizes
    # We create a logarithmic scale of box sizes from 1 pixel up to 1/5th of the image size
    max_box_size = min(Lx, Ly) // 5
    # Generate logarithmically spaced sizes and ensure they are unique integers
    sizes = np.logspace(0.1, np.log10(max_box_size), num=20, base=10)
    sizes = np.unique(np.floor(sizes)).astype(int)

    counts = []

    # 3. Count the boxes for each grid size
    for size in sizes:
        # Determine the number of boxes in X and Y directions
        Nx = int(np.ceil(Lx / size))
        Ny = int(np.ceil(Ly / size))

        # Create the grid boundaries (bins)
        xbins = np.arange(0, Nx * size + 1, size)
        ybins = np.arange(0, Ny * size + 1, size)

        # Drop the pixels into the grid using a 2D histogram
        H, _, _ = np.histogram2d(pixels[:, 0], pixels[:, 1], bins=(ybins, xbins))

        # Count how many boxes contain at least 1 pixel (H > 0)
        counts.append(np.sum(H > 0))

    # 4. Calculate the slope of the line of best fit (Fractal Dimension)
    # x-axis: log(1 / box_size)
    # y-axis: log(number of filled boxes)
    x_vals = np.log(1.0 / sizes)
    y_vals = np.log(counts)

    # np.polyfit returns [slope, intercept]
    coeffs = np.polyfit(x_vals, y_vals, 1)
    fractal_dimension = coeffs[0]

    return fractal_dimension


def calculate_skeleton_entropies(skeleton_array):
    """
    Calculates both Global and Local entropy for a binary skeleton image.
    """
    # 1. Global Binary Entropy (Effectively measures Skeleton Density)
    global_ent = shannon_entropy(skeleton_array)

    # 2. Local Spatial Entropy (Measures structural chaos)
    # Convert skeleton to an 8-bit integer array as required by skimage rank filters
    skel_8bit = (skeleton_array > 0).astype(np.uint8) * 255

    # Define the neighborhood size (a circle with a radius of 10 pixels)
    neighborhood = disk(10)

    # Calculate entropy for every local neighborhood
    # Catching warnings here because pure black regions might throw a low-contrast warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_ent_map = entropy(skel_8bit, neighborhood)

    # The final feature is the average chaos across the entire image
    mean_local_ent = np.mean(local_ent_map)

    return global_ent, mean_local_ent


def fft_metric(cropped_image):
    line_thickness = 20

    f_transform = np.fft.fft2(cropped_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    log_magnitude = np.log(np.abs(f_transform_shifted) + 1e-8)

    height, width = cropped_image.shape
    cy, cx = height // 2, width // 2
    mask_radius = 100
    percentile = 70

    y, x = np.ogrid[-cy:height - cy, -cx:width - cx]
    distance_from_center = np.sqrt(x ** 2 + y ** 2)
    min_val = np.min(log_magnitude)
    log_magnitude[distance_from_center <= mask_radius] = min_val

    threshold_value = np.percentile(log_magnitude, percentile)
    binary_fft = (log_magnitude >= threshold_value).astype(int)

    cross_mask = np.zeros_like(binary_fft)
    cross_mask[cy - line_thickness: cy + line_thickness + 1, :] = 1
    cross_mask[:, cx - line_thickness: cx + line_thickness + 1] = 1

    pixels_in_cross = np.sum(binary_fft * cross_mask)
    total_white_pixels = np.sum(binary_fft)

    return pixels_in_cross / (total_white_pixels + 1e-10)

def load_real_size(path) -> float:
    with open(path, "r", encoding="latin1", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith(r"\Scan Size:"):
                # example: \Scan Size: 92516.8 nm
                parts = line.split(":")[1].strip().split()
                return float(parts[0])  # value in nm

    assert False

def run_tests(folder):
    # loading images and filtering if metadata are available
    images: list = []
    filenames: list = []


    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(".bmp"):
            # // check  if there exists a file called filename + "_1.bmp"
            meta_path = os.path.join(folder, filename.replace("_1.bmp", ""))


            if not os.path.exists(meta_path):
                print(f"[BAD] {meta_path} doesnt exist!!")
                continue

            full_path = os.path.join(folder, filename)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                filenames.append(full_path)


    # Initialize the predictor once when your app starts
    predictor = TearPredictor(
        binary_weights_path='./stage1_healthy72_unhealthy96.pth',
        disease_weights_path='./stage2_4class_disease_model.pth',
        scaler_stage1_path='./scaler_stage1.joblib',
        scaler_stage2_path='./scaler_stage2.joblib'
    )

    warnings.filterwarnings("ignore")

    import csv

    with open("./output.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["filename", "prediction", "confidence"])

        for filename in filenames:
            try:
                if filename.lower().endswith(".bmp"):

                    if not os.path.exists(filename):
                        writer.writerow([filename, "null", "null"])
                        continue

                    prediction, confidence = predictor.predict(filename)
                    writer.writerow([filename, prediction, confidence])
            except Exception:
                writer.writerow([filename, "null", "null"])
def classify(
        input_path: str,
        want_skeleton=False,
        want_binary=False,
        want_branching=False,
        want_dots=False,
        want_components=False,
        want_fft=False
):
    # Initialize the predictor once when your app starts
    predictor = TearPredictor(
        binary_weights_path='./stage1_healthy72_unhealthy96.pth',
        disease_weights_path='./stage2_4class_disease_model.pth',
        scaler_stage1_path='./scaler_stage1.joblib',
        scaler_stage2_path='./scaler_stage2.joblib'
    )

    warnings.filterwarnings("ignore")

    img = normalize(crop_image(load_image(input_path)),input_path)
    processed, skell = process_skeleton(img)

    result = {}
    result["label"] = predictor.predict(input_path)

    if want_skeleton:
        save_image(skell, "skeleton.bmp")
    if want_binary:
        save_image(processed, "binary.bmp")
    if want_branching:
        result["branching_factor"] = branching_factor(skell)
    if want_dots:
        result["dots"] = get_spot_count(img)
    if want_components:
        result["connected_components"] = count_components(skell)
    if want_fft:
        save_image(fft_metric(img), "fft.bmp")

    return result

def main():
    parser = argparse.ArgumentParser(description="Medical image classifier")

    parser.add_argument(
        "--input-file",
        type=str,
        required=False,
        help="Path to input image"
    )

    parser.add_argument(
        "--test-folder",
        type=str,
        required=False,
        help="Path to input folder"
    )

    # feature flags
    parser.add_argument("--skeleton", action="store_true")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--branching-factor", action="store_true")
    parser.add_argument("--dots", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--fft", action="store_true")

    args = parser.parse_args()
    if args.test_folder:
        run_tests(args.test_folder)
        exit(0)
    if not args.input_file or not os.path.exists(args.input_file):
        print(f"Error: file does not exist f{args.input_file}", file=sys.stderr)
        sys.exit(1)
    try:
        result = classify(
            args.input_file,
            want_skeleton=args.skeleton,
            want_binary=args.binary,
            want_branching=args.branching_factor,
            want_dots=args.dots,
            want_components=args.components,
            want_fft=args.fft
        )

        # --- always print label ---
        print(result["label"])

        # --- optional outputs ---
        for key, value in result.items():
            if key == "label":
                continue
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
