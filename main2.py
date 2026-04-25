# ============================================================
# INFERENCIA COMPLETA TMH
# Modelo A corregido: iris automático
# Modelo B nuevo: menisco lagrimal real
# ============================================================

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp

from albumentations.pytorch import ToTensorV2


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Cambia esta ruta por tu imagen
IMAGE_PATH = "tmh4.PNG"

# Modelo A corregido para iris
MODEL_A_PATH = "model_a_fixed_iris.pth"

# Nuevo Modelo B entrenado con dataset de menisco
MODEL_B_PATH = "model_b_meniscus_new.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo A fue usado con 512x512
INPUT_SIZE_A = 512

# Nuevo Modelo B fue entrenado con 640x480
INPUT_H_B = 480
INPUT_W_B = 640

IRIS_DIAMETER_MM = 11.5

# Modelo A corregido:
# 0 = fondo
# 1 = iris
# 2 = región ocular
NUM_CLASSES_A = 3
IRIS_CLASS_ID = 1
EYE_CLASS_ID = 2

# Modelo B nuevo:
# 0 = fondo
# 1 = menisco lagrimal
THRESHOLD_B = 0.5

SAVE_PATH = "tmh_resultado_final.png"

print("Device:", DEVICE)
print("Imagen:", IMAGE_PATH)
print("Modelo A:", MODEL_A_PATH)
print("Modelo B:", MODEL_B_PATH)


# ============================================================
# 2. CARGAR MODELOS
# ============================================================

model_A = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES_A
).to(DEVICE)

model_A.load_state_dict(
    torch.load(MODEL_A_PATH, map_location=DEVICE)
)

model_A.eval()


model_B = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

model_B.load_state_dict(
    torch.load(MODEL_B_PATH, map_location=DEVICE)
)

model_B.eval()

print("✅ Modelos cargados correctamente.")


# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================

preprocess_A = A.Compose([
    A.Resize(INPUT_SIZE_A, INPUT_SIZE_A),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

preprocess_B = A.Compose([
    A.Resize(height=INPUT_H_B, width=INPUT_W_B),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


def read_image(image_path):
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def prepare_tensor_A(img_rgb):
    tensor = preprocess_A(image=img_rgb)["image"]
    tensor = tensor.unsqueeze(0).to(DEVICE)
    return tensor


def prepare_tensor_B(img_rgb):
    tensor = preprocess_B(image=img_rgb)["image"]
    tensor = tensor.unsqueeze(0).to(DEVICE)
    return tensor


# ============================================================
# 4. PREDICCIÓN MODELO A - IRIS
# ============================================================

def predict_model_A(img_rgb):
    h, w = img_rgb.shape[:2]

    tensor = prepare_tensor_A(img_rgb)

    with torch.no_grad():
        logits = model_A(tensor)

        if isinstance(logits, dict):
            if "out" in logits:
                logits = logits["out"]
            elif "logits" in logits:
                logits = logits["logits"]
            else:
                raise ValueError(f"Salida no reconocida en model_A: {logits.keys()}")

        pred = torch.argmax(logits, dim=1)
        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

    pred = cv2.resize(
        pred,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return pred


# ============================================================
# 5. PREDICCIÓN MODELO B NUEVO - MENISCO
# ============================================================

def predict_model_B(img_rgb, threshold=0.5):
    h, w = img_rgb.shape[:2]

    tensor = prepare_tensor_B(img_rgb)

    with torch.no_grad():
        logits = model_B(tensor)

        if isinstance(logits, dict):
            if "out" in logits:
                logits = logits["out"]
            elif "logits" in logits:
                logits = logits["logits"]
            else:
                raise ValueError(f"Salida no reconocida en model_B: {logits.keys()}")

        prob = torch.sigmoid(logits)
        prob = prob.squeeze().cpu().numpy()

    prob = cv2.resize(
        prob,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = (prob >= threshold).astype(np.uint8)

    return prob, mask


# ============================================================
# 6. EXTRAER DIÁMETRO DEL IRIS DESDE MODELO A
# ============================================================

def get_iris_diameter_from_model_A(img_rgb, pred_A):
    iris_mask = (pred_A == IRIS_CLASS_ID).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, kernel)
    iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(
        iris_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = [c for c in cnts if cv2.contourArea(c) > 100]

    if len(cnts) == 0:
        return None, img_rgb.copy(), {
            "error": "No se detectó iris con Modelo A corregido"
        }

    iris_cnt = max(cnts, key=cv2.contourArea)

    if len(iris_cnt) >= 5:
        ellipse = cv2.fitEllipse(iris_cnt)
        (cx, cy), (axis1, axis2), angle = ellipse

        iris_cx = int(cx)
        iris_cy = int(cy)

        # Usamos el eje menor para no inflar el diámetro.
        iris_diam_px = float(min(axis1, axis2))
    else:
        x, y, ww, hh = cv2.boundingRect(iris_cnt)
        iris_cx = x + ww // 2
        iris_cy = y + hh // 2
        iris_diam_px = float(min(ww, hh))

    if iris_diam_px < 20:
        return None, img_rgb.copy(), {
            "error": f"Iris demasiado pequeño: {iris_diam_px:.2f}px"
        }

    out = img_rgb.copy()

    cv2.drawContours(out, [iris_cnt], -1, (0, 255, 255), 2)
    cv2.circle(out, (iris_cx, iris_cy), 4, (255, 0, 0), -1)

    cv2.putText(
        out,
        f"Iris diam={iris_diam_px:.1f}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    info = {
        "iris_diam_px": iris_diam_px,
        "iris_cx": iris_cx,
        "iris_cy": iris_cy,
        "iris_area": int(cv2.contourArea(iris_cnt))
    }

    return iris_diam_px, out, info


# ============================================================
# 7. CALCULAR TMH DESDE MODELO B NUEVO + IRIS AUTOMÁTICO
# ============================================================

def calculate_tmh_from_model_b_robust(img_rgb, pred_B, iris_info):
    h, w = img_rgb.shape[:2]

    iris_diam_px = float(iris_info["iris_diam_px"])
    iris_cx = int(iris_info["iris_cx"])
    iris_cy = int(iris_info["iris_cy"])

    mask = pred_B.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ROI anatómica debajo del iris.
    # Esto evita elegir componentes lejos del iris.
    x1 = max(0, int(iris_cx - 0.60 * iris_diam_px))
    x2 = min(w, int(iris_cx + 0.60 * iris_diam_px))

    y1 = max(0, int(iris_cy + 0.05 * iris_diam_px))
    y2 = min(h, int(iris_cy + 0.90 * iris_diam_px))

    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    n_cc, labels, stats, cents = cv2.connectedComponentsWithStats(roi_mask)

    if n_cc <= 1:
        return None, img_rgb.copy(), {
            "error": "Modelo B no detectó menisco dentro de la ROI inferior del iris",
            "roi": (x1, y1, x2, y2)
        }

    candidates = []

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

        # Evita componentes demasiado gruesos para ser menisco.
        if hh > 0.30 * iris_diam_px:
            continue

        # Debe estar debajo del centro del iris.
        if cy < iris_cy:
            continue

        candidates.append({
            "idx": i,
            "area": area,
            "x": x,
            "y": y,
            "w": ww,
            "h": hh,
            "cx": cx,
            "cy": cy
        })

    if len(candidates) == 0:
        return None, img_rgb.copy(), {
            "error": "No hay componente válido de menisco después de filtros",
            "roi": (x1, y1, x2, y2)
        }

    # Elegir componente más cercano al centro inferior del iris.
    def score_candidate(c):
        dist_x = abs(c["cx"] - iris_cx) / iris_diam_px
        dist_y = abs(c["cy"] - (iris_cy + 0.45 * iris_diam_px)) / iris_diam_px
        area_bonus = -0.0003 * c["area"]
        return dist_x + dist_y + area_bonus

    best = min(candidates, key=score_candidate)
    component = (labels == best["idx"]).astype(np.uint8)

    ys, xs = np.where(component > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None, img_rgb.copy(), {
            "error": "Componente de menisco vacío después de selección"
        }

    x_left = int(xs.min())
    x_right = int(xs.max())

    column_heights = []
    column_data = []

    for x in range(x_left, x_right + 1):
        y_col = np.where(component[:, x] > 0)[0]

        if len(y_col) == 0:
            continue

        y_top = int(y_col.min())
        y_bottom = int(y_col.max())
        height = y_bottom - y_top + 1

        # Como el modelo nuevo está entrenado solo para menisco,
        # permitimos alturas moderadas, pero evitamos outliers.
        if 1 <= height <= int(0.25 * iris_diam_px):
            column_heights.append(height)
            column_data.append((x, y_top, y_bottom, height))

    if len(column_heights) == 0:
        return None, img_rgb.copy(), {
            "error": "No se pudieron medir alturas válidas del menisco"
        }

    # Medición robusta: mediana por columnas.
    tmh_px = float(np.median(column_heights))

    target_idx = int(np.argmin(np.abs(np.array(column_heights) - tmh_px)))
    x_mid, y_top_mid, y_bottom_mid, h_mid = column_data[target_idx]

    tmh_mm = round(tmh_px * IRIS_DIAMETER_MM / iris_diam_px, 3)

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

    info = {
        "tmh_px_median": round(tmh_px, 2),
        "tmh_px_min": int(min(column_heights)),
        "tmh_px_max": int(max(column_heights)),
        "iris_diam_px": round(float(iris_diam_px), 2),
        "iris_cx": iris_cx,
        "iris_cy": iris_cy,
        "tmh_mm": tmh_mm,
        "diagnosis": diagnosis,
        "meniscus_area": int(best["area"]),
        "component_width": int(best["w"]),
        "component_height": int(best["h"]),
        "roi": (x1, y1, x2, y2),
        "x_left": x_left,
        "x_right": x_right,
        "y_top_used": y_top_mid,
        "y_bottom_used": y_bottom_mid
    }

    out = img_rgb.copy()

    # Dibujar ROI
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Overlay del menisco seleccionado
    color = np.zeros_like(out)
    color[:, :, 1] = component * 255
    out = cv2.addWeighted(out, 1.0, color, 0.45, 0)

    # Línea vertical de medición
    cv2.line(out, (x_mid, y_top_mid), (x_mid, y_bottom_mid), (255, 0, 0), 2)

    # Líneas superior e inferior locales
    cv2.line(out, (x_left, y_top_mid), (x_right, y_top_mid), (0, 255, 255), 2)
    cv2.line(out, (x_left, y_bottom_mid), (x_right, y_bottom_mid), (255, 0, 255), 2)

    text = f"TMH={tmh_mm:.3f} mm | {diagnosis}"

    cv2.putText(
        out,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    return tmh_mm, out, info


# ============================================================
# 8. EJECUTAR INFERENCIA COMPLETA
# ============================================================

img_rgb = read_image(IMAGE_PATH)

# Predicción Modelo A
pred_A = predict_model_A(img_rgb)

print("\nClases predichas por Modelo A:")
print(np.unique(pred_A, return_counts=True))

iris_diam_px, iris_result, iris_info = get_iris_diameter_from_model_A(
    img_rgb,
    pred_A
)

print("\nInformación del iris:")
print(iris_info)

if iris_diam_px is None:
    raise RuntimeError("No se pudo calcular el diámetro del iris. Revisa la máscara del Modelo A.")

# Predicción Modelo B nuevo
prob_B, pred_B = predict_model_B(
    img_rgb,
    threshold=THRESHOLD_B
)

print("\nModelo B probabilidad:")
print("min:", float(prob_B.min()))
print("max:", float(prob_B.max()))
print("mean:", float(prob_B.mean()))

print("\nClases máscara Modelo B:")
print(np.unique(pred_B, return_counts=True))

# Cálculo TMH
tmh_mm, tmh_result, tmh_info = calculate_tmh_from_model_b_robust(
    img_rgb,
    pred_B,
    iris_info=iris_info
)

print("\nResultado TMH:")
print(tmh_info)

if tmh_mm is None:
    raise RuntimeError("No se pudo calcular TMH. Revisa la máscara del Modelo B.")

print("\n====================================")
print(f"TMH final: {tmh_mm:.3f} mm")
print(f"Diagnóstico: {tmh_info['diagnosis']}")
print("====================================")


# ============================================================
# 9. VISUALIZACIONES
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Imagen original")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(pred_A, cmap="tab10", vmin=0, vmax=2)
plt.colorbar()
plt.axis("off")
plt.title("Máscara Modelo A: 0=fondo, 1=iris, 2=ojo")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(iris_result)
plt.axis("off")
plt.title("Iris detectado automáticamente")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(prob_B, cmap="gray")
plt.colorbar()
plt.axis("off")
plt.title("Mapa de probabilidad Modelo B nuevo")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(pred_B, cmap="gray")
plt.axis("off")
plt.title(f"Máscara Modelo B nuevo - threshold={THRESHOLD_B}")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(tmh_result)
plt.axis("off")
plt.title("Resultado final TMH automático")
plt.show()


# ============================================================
# 10. GUARDAR RESULTADO
# ============================================================

cv2.imwrite(
    SAVE_PATH,
    cv2.cvtColor(tmh_result, cv2.COLOR_RGB2BGR)
)

print("Imagen resultado guardada en:", SAVE_PATH)