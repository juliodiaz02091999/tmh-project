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
        input_size: int = 512,
        iris_diameter_mm: float = 11.5,
        threshold_b: float = 0.7,
        device: str | None = None,
    ):
        self.model_a_path = model_a_path
        self.model_b_path = model_b_path
        self.input_size = input_size
        self.iris_diameter_mm = iris_diameter_mm
        self.threshold_b = threshold_b

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.preprocess = A.Compose(
            [
                A.Resize(self.input_size, self.input_size),
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
            activation="sigmoid",
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

    def _prepare_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        tensor = self.preprocess(image=img_rgb)["image"]
        return tensor.unsqueeze(0).to(self.device)

    def _predict_model_a(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        tensor = self._prepare_tensor(img_rgb)
        with torch.no_grad():
            logits = self.model_a(tensor)
            if isinstance(logits, dict):
                logits = logits.get("out") or logits.get("logits")
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    def _predict_model_b(self, img_rgb: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_rgb.shape[:2]
        tensor = self._prepare_tensor(img_rgb)
        with torch.no_grad():
            output = self.model_b(tensor)
            if isinstance(output, dict):
                output = output.get("out") or output.get("logits")
            prob = output
            if prob.min() < 0 or prob.max() > 1:
                prob = torch.sigmoid(prob)
            prob = prob.squeeze().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob >= threshold).astype(np.uint8)
        return prob, mask

    def _get_iris_diameter_px(self, pred_a: np.ndarray, iris_class_id: int = 1) -> float:
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
            (_, _), (axis1, axis2), _ = cv2.fitEllipse(iris_cnt)
            iris_diam_px = float(min(axis1, axis2))
        else:
            x, y, ww, hh = cv2.boundingRect(iris_cnt)
            iris_diam_px = float(min(ww, hh))
        if iris_diam_px < 20:
            raise RuntimeError(f"Iris demasiado pequeño: {iris_diam_px:.2f}px")
        return iris_diam_px

    def _calculate_tmh_mm(self, pred_b_mask: np.ndarray, iris_diam_px: float) -> TMHResult:
        mask = pred_b_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        h, w = mask.shape[:2]
        n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n_cc <= 1:
            raise RuntimeError("Modelo B no detectó menisco")

        candidates: list[dict] = []
        for i in range(1, n_cc):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = cents[i]
            if area < 20:
                continue
            if cy < h * 0.35:
                continue
            candidates.append({"idx": i, "area": area})

        if not candidates:
            raise RuntimeError("No hay componente válido de menisco")

        best = max(candidates, key=lambda d: d["area"])
        component = (labels == best["idx"]).astype(np.uint8)
        ys, xs = np.where(component > 0)
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
            if 1 <= height <= 30:
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
        iris_diam_px = self._get_iris_diameter_px(pred_a)
        _, pred_b = self._predict_model_b(img_rgb, threshold=self.threshold_b)
        return self._calculate_tmh_mm(pred_b, iris_diam_px=iris_diam_px)

