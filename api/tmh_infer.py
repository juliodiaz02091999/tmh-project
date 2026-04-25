import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2


@dataclass(frozen=True)
class TMHResult:
    tmh_mm: float
    diagnosis: str
    iris_diam_px: float
    tmh_px_median: float


class TMHInferencer:
    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        input_size_a: int = 512,
        input_h_b: int = 480,
        input_w_b: int = 640,
        iris_diameter_mm: float = 11.5,
        threshold_b: float = 0.5,
        device: str | None = None,
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.input_size_a = input_size_a
        self.input_h_b = input_h_b
        self.input_w_b = input_w_b
        self.iris_diameter_mm = iris_diameter_mm
        self.threshold_b = threshold_b

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.preprocess_a = A.Compose(
            [
                A.Resize(self.input_size_a, self.input_size_a),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.preprocess_b = A.Compose(
            [
                A.Resize(height=self.input_h_b, width=self.input_w_b),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.model_a = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=3,  # 0 fondo, 1 iris, 2 ojo
        ).to(self.device)

        self.model_b = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)

        self._load_models()

    def _load_models(self) -> None:
        if not os.path.exists(self.model_a_path):
            raise FileNotFoundError(f"MODEL_A_PATH no existe: {self.model_a_path}")
        if not os.path.exists(self.model_b_path):
            raise FileNotFoundError(f"MODEL_B_PATH no existe: {self.model_b_path}")

        self.model_a.load_state_dict(torch.load(self.model_a_path, map_location=self.device))
        self.model_b.load_state_dict(torch.load(self.model_b_path, map_location=self.device))
        self.model_a.eval()
        self.model_b.eval()

    def _prepare_tensor_a(self, img_rgb: np.ndarray) -> torch.Tensor:
        tensor = self.preprocess_a(image=img_rgb)["image"]
        return tensor.unsqueeze(0).to(self.device)

    def _prepare_tensor_b(self, img_rgb: np.ndarray) -> torch.Tensor:
        tensor = self.preprocess_b(image=img_rgb)["image"]
        return tensor.unsqueeze(0).to(self.device)

    def _predict_model_a(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        tensor = self._prepare_tensor_a(img_rgb)
        with torch.no_grad():
            logits = self.model_a(tensor)
            if isinstance(logits, dict):
                logits = logits.get("out") or logits.get("logits")
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    def _predict_model_b(self, img_rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_rgb.shape[:2]
        tensor = self._prepare_tensor_b(img_rgb)
        with torch.no_grad():
            logits = self.model_b(tensor)
            if isinstance(logits, dict):
                logits = logits.get("out") or logits.get("logits")
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob >= threshold).astype(np.uint8)
        return prob, mask

    def _get_iris_info(self, pred_a: np.ndarray, iris_class_id: int = 1) -> dict:
        iris_mask = (pred_a == iris_class_id).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, kernel)
        iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 100]
        if not cnts:
            raise RuntimeError("No se detectó iris (Modelo A)")
        iris_cnt = max(cnts, key=cv2.contourArea)
        if len(iris_cnt) >= 5:
            (cx, cy), (axis1, axis2), _ = cv2.fitEllipse(iris_cnt)
            iris_diam_px = float(min(axis1, axis2))
        else:
            x, y, ww, hh = cv2.boundingRect(iris_cnt)
            cx = x + ww / 2.0
            cy = y + hh / 2.0
            iris_diam_px = float(min(ww, hh))
        if iris_diam_px < 20:
            raise RuntimeError(f"Iris demasiado pequeño: {iris_diam_px:.2f}px")
        return {
            "iris_diam_px": float(iris_diam_px),
            "iris_cx": int(round(float(cx))),
            "iris_cy": int(round(float(cy))),
        }

    def _calculate_tmh_mm(self, pred_b_mask: np.ndarray, iris_info: dict) -> TMHResult:
        iris_diam_px = float(iris_info["iris_diam_px"])
        iris_cx = int(iris_info["iris_cx"])
        iris_cy = int(iris_info["iris_cy"])

        mask = pred_b_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        h, w = mask.shape[:2]

        # ROI anatómica debajo del iris (main2.py)
        x1 = max(0, int(iris_cx - 0.60 * iris_diam_px))
        x2 = min(w, int(iris_cx + 0.60 * iris_diam_px))
        y1 = max(0, int(iris_cy + 0.05 * iris_diam_px))
        y2 = min(h, int(iris_cy + 0.90 * iris_diam_px))

        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(roi_mask)
        if n_cc <= 1:
            raise RuntimeError("Modelo B no detectó menisco dentro de la ROI inferior del iris")

        candidates: list[dict] = []
        for i in range(1, n_cc):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = cents[i]
            if area < 10:
                continue
            if ww < 3:
                continue
            if hh < 1:
                continue
            if hh > 0.30 * iris_diam_px:
                continue
            if cy < iris_cy:
                continue
            candidates.append({"idx": i, "area": area, "x": x, "y": y, "w": ww, "h": hh, "cx": cx, "cy": cy})

        if not candidates:
            raise RuntimeError("No hay componente válido de menisco")

        def score_candidate(c: dict) -> float:
            dist_x = abs(float(c["cx"]) - iris_cx) / iris_diam_px
            dist_y = abs(float(c["cy"]) - (iris_cy + 0.45 * iris_diam_px)) / iris_diam_px
            area_bonus = -0.0003 * float(c["area"])
            return float(dist_x + dist_y + area_bonus)

        best = min(candidates, key=score_candidate)
        component = (labels == best["idx"]).astype(np.uint8)
        ys, xs = np.where(component > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("Componente de menisco vacío después de selección")
        x_left = int(xs.min())
        x_right = int(xs.max())

        column_heights: list[int] = []
        for x in range(x_left, x_right + 1):
            y_col = np.where(component[:, x] > 0)[0]
            if len(y_col) == 0:
                continue
            y_top = int(y_col.min())
            y_bottom = int(y_col.max())
            height = y_bottom - y_top + 1
            if 1 <= height <= int(0.25 * iris_diam_px):
                column_heights.append(height)

        if not column_heights:
            raise RuntimeError("No se pudieron medir alturas válidas del menisco")

        tmh_px = float(np.median(column_heights))
        tmh_mm = float(tmh_px * self.iris_diameter_mm / float(iris_diam_px))

        if tmh_mm < 0.10:
            diagnosis = "DED SEVERO"
        elif tmh_mm < 0.20:
            diagnosis = "DED MODERADO"
        elif tmh_mm < 0.30:
            diagnosis = "DED LEVE"
        elif tmh_mm <= 1.00:
            diagnosis = "NORMAL"
        else:
            diagnosis = "NO CONFIABLE"

        return TMHResult(
            tmh_mm=round(tmh_mm, 3),
            diagnosis=diagnosis,
            iris_diam_px=round(float(iris_diam_px), 2),
            tmh_px_median=round(float(tmh_px), 2),
        )

    def infer_from_bgr(self, img_bgr: np.ndarray) -> TMHResult:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Imagen inválida")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred_a = self._predict_model_a(img_rgb)
        iris_info = self._get_iris_info(pred_a)
        _, pred_b = self._predict_model_b(img_rgb, threshold=self.threshold_b)
        return self._calculate_tmh_mm(pred_b, iris_info=iris_info)

