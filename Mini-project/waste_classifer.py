import os
import sys
import warnings
import time
import pickle
import argparse
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

DATASET_DIR   = "dataset"
IMAGE_SIZE    = (256, 256)
HOG_SIZE      = (128, 128)
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

H_BINS, S_BINS, V_BINS = 8, 8, 8
HIST_RANGES = [0, 180, 0, 256, 0, 256]

GRABCUT_ITER  = 5
BORDER_FRAC  = 0.10

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

MODEL_SVM_PATH  = "model_svm.pkl"
MODEL_RF_PATH   = "model_rf.pkl"
SCALER_PATH     = "scaler.pkl"
ENCODER_PATH    = "label_encoder.pkl"
CM_SVM_PATH     = "confusion_matrix_svm.png"
CM_RF_PATH      = "confusion_matrix_rf.png"
SAMPLE_PRED_PATH = "sample_predictions.png"
ROC_CURVES_PATH  = "roc_curves.png"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# =============================================================================
# Core Pipeline: Data Loading & Preprocessing
# =============================================================================

def load_dataset(dataset_dir: str = DATASET_DIR):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' not found.\n"
            "Please download TrashNet and place it at that path."
        )

    image_paths, labels = [], []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        found = 0
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
                found += 1
        print(f"  [{class_name:>10s}]  {found:4d} images loaded")

    print(f"\n  Total : {len(image_paths)} images across {len(set(labels))} classes\n")
    return image_paths, labels

def preprocess_image(image_path: str):
    if not os.path.isfile(image_path):
        print(f"  [WARNING] File not found: {image_path}")
        return None, None

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"  [WARNING] Cannot read image: {image_path}")
        return None, None

    bgr = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    bgr_blur = cv2.GaussianBlur(bgr, (5, 5), 0)
    hsv = cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2HSV)
    return hsv, bgr_blur

def apply_grabcut(bgr_image: np.ndarray):
    h, w = bgr_image.shape[:2]
    bx, by = int(w * BORDER_FRAC), int(h * BORDER_FRAC)
    rect = (bx, by, w - 2 * bx, h - 2 * by)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask_gc   = np.zeros((h, w), np.uint8)

    try:
        cv2.grabCut(
            bgr_image, mask_gc, rect,
            bgd_model, fgd_model,
            GRABCUT_ITER, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        print(f"  [WARNING] GrabCut failed ({exc}); using full image.")
        return bgr_image, np.ones((h, w), dtype=np.uint8) * 255

    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if fg_mask.sum() < (h * w * 0.05 * 255):
        return bgr_image, np.ones((h, w), dtype=np.uint8) * 255

    fg_bgr = cv2.bitwise_and(bgr_image, bgr_image, mask=fg_mask)
    return fg_bgr, fg_mask

# =============================================================================
# Feature Extraction (Enhanced for High Accuracy)
# =============================================================================

def extract_features(fg_bgr: np.ndarray, mask: np.ndarray = None):
    # 1. HSV Histogram
    hsv_image = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, [H_BINS, S_BINS, V_BINS], HIST_RANGES)
    cv2.normalize(hsv_hist, hsv_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    hsv_feat = hsv_hist.flatten()
    
    # 2. LAB Histogram
    lab_image = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2LAB)
    lab_hist = cv2.calcHist([lab_image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(lab_hist, lab_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    lab_feat = lab_hist.flatten()

    # 3. HOG Features
    gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, HOG_SIZE)
    hog = cv2.HOGDescriptor(HOG_SIZE, (16,16), (8,8), (8,8), 9)
    hog_feat = hog.compute(gray_resized).flatten()
    
    # 4. Color Statistics (Mean & Std)
    mean_val, std_val = cv2.meanStdDev(fg_bgr, mask=mask)
    stat_feat = np.concatenate([mean_val.flatten(), std_val.flatten()]) / 255.0

    return np.concatenate([hsv_feat, lab_feat, hog_feat, stat_feat]).astype(np.float32)

def build_feature_matrix(image_paths: list, labels: list):
    X, y = [], []
    n = len(image_paths)
    t0 = time.time()

    for idx, (path, label) in enumerate(zip(image_paths, labels), 1):
        _, bgr_blur = preprocess_image(path)
        if bgr_blur is None:
            continue

        fg_bgr, fg_mask = apply_grabcut(bgr_blur)
        feat = extract_features(fg_bgr, fg_mask)

        X.append(feat)
        y.append(label)

        if idx % 50 == 0 or idx == n:
            elapsed = time.time() - t0
            print(f"  Processed {idx:4d}/{n}  ({elapsed:.1f}s elapsed)")

    return np.array(X, dtype=np.float32), y

# =============================================================================
# Model Training & Evaluation
# =============================================================================

def train_model(X_train: np.ndarray, y_train: np.ndarray):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("  Training SVM (RBF kernel, C=10) ...")
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE, probability=True)
    svm_clf.fit(X_scaled, y_enc)
    print("  SVM training complete.")

    print("  Training Random Forest (n_estimators=300) ...")
    rf_clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
    rf_clf.fit(X_scaled, y_enc)
    print("  Random Forest training complete.")

    return svm_clf, rf_clf, scaler, le

def evaluate_model(clf, X_test_scaled: np.ndarray, y_test_enc: np.ndarray,
                   le: LabelEncoder, clf_name: str = "Classifier", save_path: str = None):
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, y_pred)

    print(f"\n{'─'*60}")
    print(f"  {clf_name}  –  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'─'*60}")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test_enc, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"{clf_name}  –  Confusion Matrix\n(Accuracy: {acc*100:.2f}%)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved → {save_path}")
    plt.close(fig)
    return acc, y_pred

def plot_roc_curves(clf, X_test_scaled, y_test_enc, le, save_path=ROC_CURVES_PATH):
    n_classes = len(le.classes_)
    y_test_bin = label_binarize(y_test_enc, classes=list(range(n_classes)))
    y_score = clf.predict_proba(X_test_scaled)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    colours = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (cls_name, colour) in enumerate(zip(le.classes_, colours)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax = axes[i]
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"ROC  –  {cls_name}", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("One-vs-Rest ROC Curves", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ROC curves saved → {save_path}")
    plt.close(fig)

def run_cross_validation(clf, X_scaled, y_enc, cv: int = 5, clf_name="SVM"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X_scaled, y_enc, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"\n  {clf_name}  –  {cv}-Fold Cross-Validation")
    print(f"  Fold accuracies  : {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean ± Std       : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

def save_models(svm_clf, rf_clf, scaler, le):
    for obj, path in [(svm_clf, MODEL_SVM_PATH), (rf_clf, MODEL_RF_PATH), (scaler, SCALER_PATH), (le, ENCODER_PATH)]:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"\n  Models saved to disk.")

def load_models(use_rf=False):
    model_path = MODEL_RF_PATH if use_rf else MODEL_SVM_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model {model_path} not found. Train first.")
    with open(model_path, "rb") as f: clf = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le = pickle.load(f)
    return clf, scaler, le

# =============================================================================
# Visualisation & Prediction Utilities
# =============================================================================

def predict_waste(image_path: str, use_rf=False):
    clf, scaler, le = load_models(use_rf)
    _, bgr_blur = preprocess_image(image_path)
    if bgr_blur is None:
        return "Error"

    fg_bgr, fg_mask = apply_grabcut(bgr_blur)
    feat = extract_features(fg_bgr, fg_mask).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    pred_enc = clf.predict(feat_scaled)[0]
    pred_class = le.inverse_transform([pred_enc])[0]
    print(f"\n  Image: {image_path}  |  Predicted: {pred_class.upper()}")
    return pred_class

def visualise_sample_predictions(image_paths, labels, clf, scaler, le, n_samples: int = 12, save_path=SAMPLE_PRED_PATH):
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(image_paths), size=min(n_samples, len(image_paths)), replace=False)
    cols = 4
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for plot_idx, img_idx in enumerate(indices):
        path, true_label = image_paths[img_idx], labels[img_idx]
        _, bgr_blur = preprocess_image(path)
        if bgr_blur is None:
            continue
        fg_bgr, fg_mask = apply_grabcut(bgr_blur)
        feat = extract_features(fg_bgr, fg_mask).reshape(1, -1)
        pred_enc = clf.predict(scaler.transform(feat))[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        ax = axes[plot_idx]
        ax.imshow(cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        colour = "green" if pred_label == true_label else "red"
        ax.set_title(f"True : {true_label}\nPred : {pred_label}", fontsize=8, color=colour, pad=3)

    for ax in axes[len(indices):]: ax.set_visible(False)
    fig.suptitle("Sample Predictions – green = correct, red = wrong", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run_realtime_demo(use_rf=False, camera_index=0):
    clf, scaler, le = load_models(use_rf)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        return

    print("\n  Real-time demo started. Press 'q' to quit.\n")
    class_colours = {"cardboard":(0,165,255), "glass":(255,255,0), "metal":(128,128,128), "paper":(0,255,0), "plastic":(255,0,255), "trash":(0,0,255)}
    pred_label, frame_count = "—", 0

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame_count += 1
        if frame_count % 15 == 0:
            try:
                small = cv2.resize(frame, IMAGE_SIZE)
                blurred = cv2.GaussianBlur(small, (5, 5), 0)
                fg_bgr, fg_mask = apply_grabcut(blurred)
                feat = extract_features(fg_bgr, fg_mask).reshape(1, -1)
                pred_label = le.inverse_transform([clf.predict(scaler.transform(feat))[0]])[0]
            except Exception as e:
                pred_label = "Error"

        h_fr, w_fr = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h_fr - 60), (w_fr, h_fr), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Waste Class:  {pred_label.upper()}", (15, h_fr - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, class_colours.get(pred_label, (255,255,255)), 2)
        cv2.putText(frame, "Press 'q' to quit", (w_fr - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow("Smart Recycling Bin", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

def plot_class_histograms():
    print("Plotting Class Hue Histograms ...")
    image_paths, labels = load_dataset()
    rng = np.random.default_rng(RANDOM_STATE)
    class_paths = defaultdict(list)
    for p, l in zip(image_paths, labels): class_paths[l].append(p)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cls_name, colour in zip(axes.flatten(), CLASSES, plt.cm.tab10(np.linspace(0, 1, 6))):
        paths = class_paths.get(cls_name, [])
        if not paths:
            ax.set_visible(False)
            continue
        chosen = rng.choice(paths, size=min(30, len(paths)), replace=False)
        hue_hists = []
        for p in chosen:
            _, bgr_blur = preprocess_image(p)
            if bgr_blur is None: continue
            fg_bgr, fg_mask = apply_grabcut(bgr_blur)
            hsv_image = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_image], [0], fg_mask, [180], [0, 180])
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            hue_hists.append(hist.flatten())

        if not hue_hists:
            ax.set_visible(False)
            continue
        mean_hist = np.mean(hue_hists, axis=0)
        hue_centres = np.linspace(0, 180, 180, endpoint=False)
        ax.plot(hue_centres, mean_hist, color=colour)
        ax.fill_between(hue_centres, 0, mean_hist, alpha=0.5, color=colour)
        ax.set_title(cls_name.title())
    plt.tight_layout()
    plt.savefig("class_histograms.png")
    plt.close(fig)

def visualise_grabcut(n_images=6):
    print("Visualising GrabCut ...")
    image_paths, labels = load_dataset()
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(image_paths), size=min(n_images, len(image_paths)), replace=False)
    fig, axes = plt.subplots(n_images, 3, figsize=(10, n_images * 2.8))
    for row, idx in enumerate(indices):
        _, bgr_blur = preprocess_image(image_paths[idx])
        fg_bgr, fg_mask = apply_grabcut(bgr_blur)
        axes[row, 0].imshow(cv2.cvtColor(bgr_blur, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(fg_mask, cmap="gray")
        axes[row, 2].imshow(cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2RGB))
        for col in range(3): axes[row, col].axis("off")
        axes[row, 0].set_title("Original" if row==0 else "")
        axes[row, 1].set_title("Mask" if row==0 else "")
        axes[row, 2].set_title("Foreground" if row==0 else "")
    plt.tight_layout()
    plt.savefig("grabcut_demo.png")
    plt.close(fig)

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Waste Classifier")
    parser.add_argument("--demo", action="store_true", help="Run realtime webcam demo")
    parser.add_argument("--predict", type=str, help="Predict a single image path")
    parser.add_argument("--analyze-features", action="store_true", help="Plot Hue histograms")
    parser.add_argument("--demo-grabcut", action="store_true", help="Demo GrabCut segmentation")
    parser.add_argument("--cross-validate", action="store_true", help="Run K-Fold CV")
    args = parser.parse_args()

    if args.demo:
        run_realtime_demo()
        return
    if args.predict:
        predict_waste(args.predict)
        return
    if args.analyze_features:
        plot_class_histograms()
        return
    if args.demo_grabcut:
        visualise_grabcut()
        return

    print("=" * 65)
    print("   Waste Item Classification for Smart Recycling Bins")
    print("   Unified High-Accuracy Traditional CV Pipeline")
    print("=" * 65)

    print("\n[1/6] Loading dataset ...")
    image_paths, labels = load_dataset(DATASET_DIR)
    if not image_paths:
        sys.exit(1)

    print("\n[2-4] Extracting Enhanced Features (HSV + LAB + HOG + Stats) ...")
    X, y = build_feature_matrix(image_paths, labels)

    print("\n[5] Splitting dataset ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    print("\n[6] Training models ...")
    svm_clf, rf_clf, scaler, le = train_model(X_train, y_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    y_test_enc     = le.transform(y_test)

    print("\n[7] Evaluating Models ...")
    evaluate_model(svm_clf, X_test_scaled, y_test_enc, le, "SVM (RBF)", CM_SVM_PATH)
    evaluate_model(rf_clf, X_test_scaled, y_test_enc, le, "Random Forest", CM_RF_PATH)

    print("\n[8] Generating ROC Curves ...")
    plot_roc_curves(rf_clf, X_test_scaled, y_test_enc, le, ROC_CURVES_PATH)

    save_models(svm_clf, rf_clf, scaler, le)
    visualise_sample_predictions(image_paths, labels, rf_clf, scaler, le)

    if args.cross_validate:
        run_cross_validation(rf_clf, scaler.transform(X), le.transform(y), cv=5, clf_name="Random Forest")

    print("\n[✓] Pipeline complete! All utilities are now unified in waste_classifier.py.")

if __name__ == "__main__":
    main()
