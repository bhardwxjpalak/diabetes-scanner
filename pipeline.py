import os
import base64
import cv2
import numpy as np
import onnxruntime as ort
import xgboost as xgb
from scipy.ndimage import label as ndlabel
from skimage.morphology import skeletonize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Paths
ROI_ONNX_PATH = os.path.join(MODELS_DIR, "ghostnet_roi_segmentation.onnx")
VESSEL_ONNX_PATH = os.path.join(MODELS_DIR, "ghostnet_vessel_segmentation.onnx")
XGB_JSON_PATH = os.path.join(MODELS_DIR, "frozen_xgb_4feat.json")
# Standardizer constants extracted from frozen StandardScaler
SCALER_MEANS = np.array([329.059649122807, 95.66816085726683, 0.8499133983694536, 64.13531100765384])
SCALER_SCALES = np.array([227.61344053752006, 56.065201873146215, 0.17529670856766813, 72.56871416835884])

# Frozen SHAP CVHI weights
W_TVL = 0.43676496
W_MBA = 0.30481678
W_FD  = 0.03998948
W_LAC = 0.21842882

DECISION_THRESHOLD = 0.53

class PipelineService:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print("[*] Initializing ONNX Runtime Sessions...")
        self.roi_session = ort.InferenceSession(ROI_ONNX_PATH, opts, providers=["CPUExecutionProvider"])
        self.vessel_session = ort.InferenceSession(VESSEL_ONNX_PATH, opts, providers=["CPUExecutionProvider"])

        print("[*] Loading Frozen XGBoost Classifier...")
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(XGB_JSON_PATH)

    def check_quality(self, img_rgb: np.ndarray, blur_threshold: float = 25.0) -> dict:
        """Stage 1: Image Quality Assessment."""
        h, w, c = img_rgb.shape
        total_pixels = h * w
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Motion blur (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Illumination & Contrast
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))

        # 3. Specular glare
        glare_mask = (img_rgb[:, :, 0] > 250) & (img_rgb[:, :, 1] > 250) & (img_rgb[:, :, 2] > 250)
        glare_ratio = float(np.sum(glare_mask) / total_pixels)

        if laplacian_var < blur_threshold:
            return {
                "pass": False,
                "reason": f"Image is blurry (Laplacian variance {laplacian_var:.1f} < {blur_threshold}). Please hold camera steady and tap to focus.",
                "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
            }
        if mean_intensity < 45.0:
            return {
                "pass": False,
                "reason": f"Image is too dark (mean intensity {mean_intensity:.1f} < 45.0). Enable flash or move to better lighting.",
                "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
            }
        if mean_intensity > 220.0:
            return {
                "pass": False,
                "reason": f"Image is overexposed (mean intensity {mean_intensity:.1f} > 220.0). Avoid direct reflection.",
                "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
            }
        if std_intensity < 15.0:
            return {
                "pass": False,
                "reason": f"Image contrast is too low (std {std_intensity:.1f} < 15.0). Ensure the sclera is centered.",
                "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
            }
        if glare_ratio > 0.05:
            return {
                "pass": False,
                "reason": f"Specular glare detected ({glare_ratio*100:.1f}% > 5.0%). Tilt camera slightly to eliminate hotspot.",
                "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
            }

        return {
            "pass": True,
            "reason": "Quality check passed.",
            "metrics": {"blur": laplacian_var, "mean": mean_intensity, "contrast": std_intensity, "glare": glare_ratio}
        }

    def preprocess_roi(self, img_rgb: np.ndarray) -> np.ndarray:
        """Stage 2A: LAB CLAHE on L channel, 256x256, RGB float32 [1, 3, 256, 256]."""
        resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        norm = enhanced_rgb.astype(np.float32) / 255.0
        # NCHW
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def preprocess_vessel(self, img_rgb: np.ndarray) -> np.ndarray:
        """Stage 2B: Green channel CLAHE + Bitwise Invert, 512x512, float32 [1, 1, 512, 512]."""
        resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        green = resized[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_g = clahe.apply(green)
        inv_g = cv2.bitwise_not(clahe_g)
        norm = inv_g.astype(np.float32) / 255.0
        # NCHW
        tensor = norm[np.newaxis, np.newaxis, ...]
        return tensor

    def segment_roi(self, roi_tensor: np.ndarray) -> np.ndarray:
        """Stage 3: GhostNet ROI segmentation."""
        input_name = self.roi_session.get_inputs()[0].name
        preds = self.roi_session.run(None, {input_name: roi_tensor})[0]
        # output shape is [1, 1, 256, 256]
        prob_map = preds[0, 0]
        binary_mask = (prob_map >= 0.50).astype(np.uint8) * 255
        return binary_mask

    def segment_vessel(self, vessel_tensor: np.ndarray, roi_mask_256: np.ndarray) -> np.ndarray:
        """Stage 4: GhostNet Vessel segmentation with ROI constraint."""
        input_name = self.vessel_session.get_inputs()[0].name
        preds = self.vessel_session.run(None, {input_name: vessel_tensor})[0]
        # output shape is [1, 1, 512, 512]
        prob_map = preds[0, 0]

        scaled_roi = cv2.resize(roi_mask_256, (512, 512), interpolation=cv2.INTER_NEAREST)
        raw_mask = (prob_map >= 0.35) & (scaled_roi > 127)

        # Remove small components < 50 pixels
        labeled, num_features = ndlabel(raw_mask)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())
            mask_sizes = sizes >= 50
            mask_sizes[0] = 0
            clean_mask = mask_sizes[labeled]
        else:
            clean_mask = raw_mask

        return clean_mask.astype(np.uint8) * 255

    def extract_biomarkers(self, vessel_mask: np.ndarray) -> dict:
        """Stage 5: 4-Biomarker extraction."""
        bin_vessel = vessel_mask > 127
        if np.count_nonzero(bin_vessel) == 0:
            return {"tvl": 0.0, "mba": 0.0, "lac": 0.0, "fd": 0.0, "skeleton": np.zeros_like(vessel_mask)}

        # Centerline skeleton
        skeleton = skeletonize(bin_vessel)

        # 1. Total Vessel Length (TVL)
        tvl = float(np.count_nonzero(skeleton))

        # 2. Mean Branching Angle (MBA)
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0.0
        nc = cv2.filter2D(skeleton.astype(np.uint8), ddepth=-1, kernel=kernel)
        nc[~skeleton] = 0
        bp_coords = np.argwhere(nc >= 3)

        angles = []
        for r, c in bp_coords:
            r0, r1 = max(0, r - 3), min(skeleton.shape[0], r + 4)
            c0, c1 = max(0, c - 3), min(skeleton.shape[1], c + 4)
            patch = skeleton[r0:r1, c0:c1].copy()
            pr, pc = r - r0, c - c0
            patch[pr, pc] = False
            labeled_patch, num_arms = ndlabel(patch, structure=np.ones((3, 3)))
            if num_arms < 2:
                continue

            arm_vectors = []
            for i in range(1, num_arms + 1):
                coords = np.argwhere(labeled_patch == i)
                if len(coords) == 0:
                    continue
                dists = np.sum((coords - np.array([pr, pc])) ** 2, axis=1)
                furthest = coords[np.argmax(dists)]
                vec = furthest - np.array([pr, pc])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    arm_vectors.append(vec / norm)

            if len(arm_vectors) >= 2:
                pair_angles = []
                for i in range(len(arm_vectors)):
                    for j in range(i + 1, len(arm_vectors)):
                        dot = np.clip(np.dot(arm_vectors[i], arm_vectors[j]), -1.0, 1.0)
                        pair_angles.append(np.arccos(dot) * 180.0 / np.pi)
                if len(pair_angles) > 0:
                    angles.append(np.mean(pair_angles))

        mba = float(np.mean(angles)) if len(angles) > 0 else 0.0

        # 3. Lacunarity (LAC) via Integral Image
        box_sizes = [8, 16, 32, 64]
        lac_vals = []
        int_img = cv2.integral(bin_vessel.astype(np.uint8))
        H, W = bin_vessel.shape
        for s in box_sizes:
            if s > H or s > W:
                continue
            mass = int_img[s:H+1, s:W+1] - int_img[0:H-s+1, s:W+1] - int_img[s:H+1, 0:W-s+1] + int_img[0:H-s+1, 0:W-s+1]
            if mass.size == 0:
                continue
            mean_m = np.mean(mass)
            if mean_m == 0:
                continue
            mean_sq = np.mean(mass ** 2)
            lac_vals.append(mean_sq / (mean_m ** 2))

        lac = float(np.mean(lac_vals)) if len(lac_vals) > 0 else 0.0

        # 4. Fractal Dimension (FD)
        box_sizes_fd = [2, 4, 8, 16, 32, 64, 128]
        counts, inv_sizes = [], []
        for s in box_sizes_fd:
            n_r = int(np.ceil(H / s))
            n_c = int(np.ceil(W / s))
            cnt = 0
            for r in range(n_r):
                for c in range(n_c):
                    if np.any(skeleton[r*s : (r+1)*s, c*s : (c+1)*s]):
                        cnt += 1
            if cnt > 0:
                counts.append(cnt)
                inv_sizes.append(1.0 / s)

        if len(counts) >= 2:
            coeffs = np.polyfit(np.log(inv_sizes), np.log(counts), deg=1)
            fd = float(np.clip(coeffs[0], 0.0, 2.0))
        else:
            fd = 0.0

        return {
            "tvl": tvl,
            "mba": mba,
            "lac": lac,
            "fd": fd,
            "skeleton": (skeleton.astype(np.uint8) * 255)
        }

    def classify(self, biomarkers: dict) -> dict:
        """Stage 6: Z-Score standardization, XGBoost inference, and CVHI score."""
        raw_vec = np.array([biomarkers["tvl"], biomarkers["mba"], biomarkers["fd"], biomarkers["lac"]])
        z_features = (raw_vec - SCALER_MEANS) / SCALER_SCALES

        dmat = xgb.DMatrix(z_features.reshape(1, -1))
        prob = float(self.xgb_model.predict(dmat)[0])

        risk_level = "DIABETES_RISK" if prob >= DECISION_THRESHOLD else "HEALTHY"

        # CVHI calculation
        raw_cvhi = (W_TVL * z_features[0] +
                    W_MBA * z_features[1] +
                    W_FD  * z_features[2] -
                    W_LAC * z_features[3])

        cvhi_score = float(np.clip((raw_cvhi + 2.5) / 5.0 * 100.0, 0.0, 100.0))

        return {
            "risk_level": risk_level,
            "probability": prob,
            "cvhi_score": cvhi_score,
            "threshold": DECISION_THRESHOLD,
            "z_features": {
                "tvl": float(z_features[0]),
                "mba": float(z_features[1]),
                "fd": float(z_features[2]),
                "lac": float(z_features[3])
            }
        }

    def run_pipeline(self, img_bytes: bytes) -> dict:
        """End-to-end pipeline execution from raw image bytes."""
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image from provided payload.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: Quality Check
        quality = self.check_quality(img_rgb)
        if not quality["pass"]:
            return {
                "success": False,
                "stage": "quality_check",
                "error": quality["reason"],
                "metrics": quality["metrics"]
            }

        # Stage 2: Preprocessing
        roi_tensor = self.preprocess_roi(img_rgb)
        vessel_tensor = self.preprocess_vessel(img_rgb)

        # Stage 3: ROI Segmentation
        roi_mask = self.segment_roi(roi_tensor)

        # Stage 4: Vessel Segmentation
        vessel_mask = self.segment_vessel(vessel_tensor, roi_mask)

        # Stage 5: Feature Extraction
        biomarkers = self.extract_biomarkers(vessel_mask)

        # Stage 6: Classification
        triage = self.classify(biomarkers)

        # Overlays for clinical visualization
        resized_orig = cv2.resize(img_rgb, (512, 512))
        scaled_roi = cv2.resize(roi_mask, (512, 512))

        # 1. ROI overlay: translucent blue
        roi_overlay = resized_orig.copy()
        roi_indices = scaled_roi > 127
        roi_overlay[roi_indices] = (0.6 * roi_overlay[roi_indices] + 0.4 * np.array([41, 121, 255])).astype(np.uint8)

        # 2. Vessel overlay: bright red on original image
        vessel_overlay = resized_orig.copy()
        vessel_indices = vessel_mask > 127
        vessel_overlay[vessel_indices] = [255, 30, 30]

        def to_b64(cv_img):
            _, buf = cv2.imencode('.png', cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
            return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

        return {
            "success": True,
            "quality": quality,
            "triage": triage,
            "biomarkers": {
                "tvl": biomarkers["tvl"],
                "mba": biomarkers["mba"],
                "lac": biomarkers["lac"],
                "fd": biomarkers["fd"]
            },
            "visualizations": {
                "roi_mask": to_b64(scaled_roi),
                "roi_overlay": to_b64(roi_overlay),
                "vessel_mask": to_b64(vessel_mask),
                "vessel_overlay": to_b64(vessel_overlay),
                "skeleton": to_b64(biomarkers["skeleton"])
            }
        }
