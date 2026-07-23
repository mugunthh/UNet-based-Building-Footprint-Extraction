# =========================
# 1. Import Libraries
# =========================

import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, concatenate
from scipy import ndimage
import json
import os

# Optional: rasterio for GeoTIFF coordinate conversion (install if needed)
try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# =========================
# 2. Model Configuration
# =========================

IMAGE_SIZE       = 512          # Tile size expected by the model
OVERLAP          = 196           # Overlap between adjacent tiles (handles border artifacts)
MORPH_KERNEL     = 3            # Kernel size for morphological close/open operations
APPROX_EPSILON   = 0.8          # Douglas-Peucker epsilon for polygon simplification (pixels)
MAX_ASPECT       = 3.0          # Max width/height ratio -- filters vehicles/elongated objects
BATCH_SIZE       = 8            # Number of tiles per inference batch (tune to GPU VRAM)

# -----------------------------------------------------------------------
# Screenshot-specific settings
# -----------------------------------------------------------------------
# Aggressive downscaling (e.g. 0.75x) makes already-small screenshot
# buildings invisible to the model.  Safe range: [1.0, 1.25] only.
INFERENCE_SCALES = [0.9, 1.0, 1.15]

# Screenshot artifacts produce many small false-positive blobs.
# Raise MIN_AREA relative to aerial defaults to suppress them.
MIN_AREA         = 500        # px  (aerial default was 500)

# Slight upscale applied to the full image before tiling so that
# buildings occupy more pixels -- closer to training-data density.
SCREENSHOT_UPSCALE = 1.3        # set to 1.0 to disable

OUTPUT_GEOJSON   = "building_footprints.geojson"
OUTPUT_MASK      = "building_mask.png"
OUTPUT_PROB_MAP  = "probability_map.png"


# =========================
# 3. Encoder Block
# =========================

class EncoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate=None, pooling=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate    = rate
        self.pooling = pooling

        self.conv1 = Conv2D(filters, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(filters, 3, padding='same', activation='relu')

        if pooling:
            self.pool = MaxPool2D((2, 2))
        if rate is not None:
            self.drop = Dropout(rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        if self.rate is not None:
            x = self.drop(x)
        x = self.conv2(x)
        if self.pooling:
            y = self.pool(x)
            return y, x
        return x


# =========================
# 4. Decoder Block
# =========================

class DecoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate=None, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rate    = rate
        self.axis    = axis

        self.convT = Conv2DTranspose(filters, 3, strides=2, padding='same')
        self.conv1 = Conv2D(filters, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(filters, 3, activation='relu', padding='same')

        if rate is not None:
            self.drop = Dropout(rate)

    def call(self, inputs):
        X, short_X = inputs
        x = self.convT(X)
        x = concatenate([x, short_X], axis=self.axis)
        x = self.conv1(x)
        if self.rate is not None:
            x = self.drop(x)
        x = self.conv2(x)
        return x


# =========================
# 5. Load Trained Model
# =========================

def load_model(model_path: str):
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "EncoderBlock": EncoderBlock,
            "DecoderBlock": DecoderBlock,
        },
    )
    print("Model loaded successfully.")
    model.summary()
    return model


# =========================
# 6. Screenshot Preprocessing
# =========================

def preprocess_screenshot(image: np.ndarray,
                           upscale: float = SCREENSHOT_UPSCALE) -> np.ndarray:
    """
    Adapt a web-screenshot satellite image so its visual distribution
    more closely matches the aerial orthophotos used during training.

    Steps applied in order
    ----------------------
    1. Spatial upscaling
       Screenshots compress building roofs into very few pixels.
       Upscaling by ~1.3x raises roof-pixel density closer to what the
       model's convolutional filters were trained on, producing stronger
       probability responses.

    2. Gaussian blur (sigma = 0.8)
       Screenshots often contain JPEG block artifacts, sharpening halos,
       and renderer anti-aliasing that create high-frequency edges.
       Those edges fire the same conv filters as real roof boundaries,
       causing false activations around text labels, road lines, and UI
       elements.  A mild blur suppresses them without softening true
       roof edges at the upscaled resolution.

    3. Brightness / contrast normalization (per-channel linear stretch)
       The model was trained on radiometrically calibrated sensor data
       with a specific intensity distribution.  Screenshots are tone-
       mapped by the browser/map renderer and may be darker, washed-out,
       or over-saturated.  Stretching each channel to [0, 1] brings the
       histogram closer to training conditions, stabilizing the prob map.

    4. Luminance equalization (CLAHE on L channel in LAB space)
       Even after linear normalization, local contrast around roofs can
       be low (dark roofs on dark shadows, light roofs on bright roads).
       CLAHE boosts local contrast adaptively so roofs stand out from
       surrounding surfaces without introducing global wash-out.

    Parameters
    ----------
    image   : float32 RGB array of shape (H, W, 3), values in [0, 1]
    upscale : spatial scale factor applied before tiling

    Returns
    -------
    preprocessed float32 RGB array, values in [0, 1]
    """

    # ── Step 1: Spatial upscaling ────────────────────────────────────────
    if upscale != 1.0:
        H, W     = image.shape[:2]
        new_H    = int(H * upscale)
        new_W    = int(W * upscale)
        image    = cv2.resize(image, (new_W, new_H),
                              interpolation=cv2.INTER_CUBIC)
        print(f"  Upscaled {W}x{H} -> {new_W}x{new_H} "
              f"(factor {upscale}x)")

    # ── Step 2: Gaussian blur to reduce screenshot artifacts ─────────────
    # ksize must be odd; sigma=0.8 is gentle but effective on JPEG noise
    image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.8, sigmaY=0.6)

    # ── Step 3: Per-channel brightness / contrast normalization ──────────
    # Linear stretch: maps [channel_min, channel_max] -> [0, 1]
    # Clips extreme outlier pixels (top/bottom 0.5 %) before stretching
    # so a single blown-out or dead pixel doesn't suppress everything else.
    normalized = np.empty_like(image)
    for c in range(3):
        ch       = image[:, :, c]
        lo       = np.percentile(ch, 0.5)
        hi       = np.percentile(ch, 99.5)
        if hi > lo:
            normalized[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        else:
            normalized[:, :, c] = ch          # flat channel, leave as-is
    image = normalized

    # ── Step 4: Luminance equalization (CLAHE in LAB) ────────────────────
    # Convert to uint8 LAB, equalize only L (lightness), convert back.
    # clipLimit=2.0 and tileGridSize=(8,8) are standard CLAHE defaults
    # that boost local contrast without over-amplifying noise.
    img_uint8 = (image * 255).astype(np.uint8)
    lab       = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image     = equalized.astype(np.float32) / 255.0

    print("  Screenshot preprocessing complete "
          "(blur + normalization + CLAHE).")
    return image


# =========================
# 7. TTA -- Test-Time Augmentation
# =========================

def tta_predict(model, tile_batch: np.ndarray) -> np.ndarray:
    """
    Run inference on a batch of tiles using 8 augmentations:
        - Original
        - Horizontal flip  (left-right)
        - Vertical flip    (up-down)
        - 90 degree rotation

    Each augmentation is reversed before averaging so all predictions
    are in the original tile's coordinate frame.

    Parameters
    ----------
    model      : loaded Keras model
    tile_batch : float32 array of shape (B, H, W, 3)

    Returns
    -------
    averaged probability map of shape (B, H, W)
    """
    def _predict(batch):
        return model.predict(batch, verbose=0)[..., 0]

    pred_orig = _predict(tile_batch)

    # flips
    flipped_h = np.flip(tile_batch, axis=2)
    pred_h = np.flip(_predict(flipped_h), axis=2)

    flipped_v = np.flip(tile_batch, axis=1)
    pred_v = np.flip(_predict(flipped_v), axis=1)

    # rotations
    rot90 = np.rot90(tile_batch, k=1, axes=(1,2))
    pred_r90 = np.rot90(_predict(rot90), k=-1, axes=(1,2))

    rot180 = np.rot90(tile_batch, k=2, axes=(1,2))
    pred_r180 = np.rot90(_predict(rot180), k=-2, axes=(1,2))

    rot270 = np.rot90(tile_batch, k=3, axes=(1,2))
    pred_r270 = np.rot90(_predict(rot270), k=-3, axes=(1,2))

    # flipped rotations
    rot90_flip = np.flip(rot90, axis=2)
    pred_r90_flip = np.flip(np.rot90(_predict(rot90_flip), k=-1, axes=(1,2)), axis=2)

    rot270_flip = np.flip(rot270, axis=2)
    pred_r270_flip = np.flip(np.rot90(_predict(rot270_flip), k=-3, axes=(1,2)), axis=2)

    return (
        pred_orig +
        pred_h +
        pred_v +
        pred_r90 +
        pred_r180 +
        pred_r270 +
        pred_r90_flip +
        pred_r270_flip
    ) / 8.0
   
# =========================
# 8. Multi-Scale Inference
# =========================

def multiscale_predict(model, tile_batch: np.ndarray,
                        scales: list = None) -> np.ndarray:
    """
    Run TTA inference at multiple scales and average results.

    For screenshot imagery INFERENCE_SCALES excludes 0.75x because
    shrinking already-small buildings makes them undetectable.

    Parameters
    ----------
    model      : loaded Keras model
    tile_batch : float32 array of shape (B, IMAGE_SIZE, IMAGE_SIZE, 3)
    scales     : list of float scale factors

    Returns
    -------
    averaged probability map of shape (B, IMAGE_SIZE, IMAGE_SIZE)
    """
    if scales is None:
        scales = INFERENCE_SCALES

    B     = tile_batch.shape[0]
    accum = np.zeros((B, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for scale in scales:
        target = int(IMAGE_SIZE * scale)

        # Resize to target, run TTA, resize predictions back
        scaled_batch = np.array([
            cv2.resize(
                cv2.resize(tile, (target, target), interpolation=cv2.INTER_LINEAR),
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR
            )
            for tile in tile_batch
        ], dtype=np.float32)

        preds_scaled = tta_predict(model, scaled_batch)

        preds_back = preds_scaled

        accum += preds_back

    return (accum / len(scales)).astype(np.float32)


# =========================
# 9. Tiled Prediction
# =========================

def tile_predict(model, image: np.ndarray) -> np.ndarray:
    """
    Splits a large satellite image into overlapping 512x512 tiles,
    runs multi-scale + TTA inference on each batch, and stitches a
    full-resolution probability map using Gaussian blending windows
    to eliminate hard seams at tile boundaries.

    Parameters
    ----------
    model  : loaded Keras model
    image  : float32 RGB array of shape (H, W, 3), values in [0, 1]

    Returns
    -------
    prob_map : float32 array of shape (H, W) with per-pixel probabilities
    """
    H, W = image.shape[:2]
    step = IMAGE_SIZE - OVERLAP

    # Gaussian blending window (down-weights tile edges)
    sigma  = IMAGE_SIZE / 6.0
    ax     = np.arange(IMAGE_SIZE) - IMAGE_SIZE / 2.0
    gauss1 = np.exp(-ax**2 / (2 * sigma**2))
    window = np.outer(gauss1, gauss1).astype(np.float32)

    # Pad image so every tile is exactly IMAGE_SIZE x IMAGE_SIZE
    pad_h  = (IMAGE_SIZE - H % step) % step if H % step != 0 else 0
    pad_w  = (IMAGE_SIZE - W % step) % step if W % step != 0 else 0
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    pH, pW = padded.shape[:2]

    coords = []
    tiles  = []
    y0 = 0
    while y0 + IMAGE_SIZE <= pH:
        x0 = 0
        while x0 + IMAGE_SIZE <= pW:
            tiles.append(padded[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE])
            coords.append((y0, x0))
            x0 += step
        y0 += step

    tiles       = np.array(tiles, dtype=np.float32)
    preds       = []
    total_tiles = len(tiles)

    for i in range(0, total_tiles, BATCH_SIZE):
        batch = tiles[i:i + BATCH_SIZE]
        pred  = multiscale_predict(model, batch, INFERENCE_SCALES)
        preds.append(pred)
        print(f"  Processed {min(i + BATCH_SIZE, total_tiles)}/{total_tiles} tiles ...",
              end='\r')

    print()
    preds = np.concatenate(preds, axis=0)

    pad_prob   = np.zeros((pH, pW), dtype=np.float32)
    pad_weight = np.zeros((pH, pW), dtype=np.float32)

    for (y0, x0), pred_tile in zip(coords, preds):
        pad_prob  [y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE] += pred_tile * window
        pad_weight[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE] += window

    pad_weight = np.where(pad_weight == 0, 1e-6, pad_weight)
    prob_map   = (pad_prob / pad_weight)[:H, :W]

    print(f"Tiled prediction complete -- {len(tiles)} tiles, "
          f"{len(INFERENCE_SCALES)} scales x 8 TTA augmentations.")
    return prob_map


# =========================
# 10. Adaptive Threshold (Otsu)
# =========================

def adaptive_threshold(prob_map: np.ndarray):
    """
    Compute a per-image threshold using Otsu's method instead of a
    hard-coded constant. Works well across varying lighting conditions
    and building densities.

    Parameters
    ----------
    prob_map : float32 probability map in [0, 1]

    Returns
    -------
    binary : uint8 array with values 0 or 1
    thresh : float threshold value that was chosen
    """
    prob_map = np.power(prob_map,0.75)
    lap = cv2.Laplacian(prob_map,cv2.CV_32F)
    prob_map = np.clip(prob_map - 0.3 * lap,0,1)
    prob_map = cv2.bilateralFilter(prob_map, 5, 0.1, 5)
    prob_map = cv2.GaussianBlur(prob_map,(5,5),0)
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    otsu_val, binary = cv2.threshold(
        prob_uint8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = otsu_val / 255.0
    
    thresh = min(0.55, max(otsu_val/255.0, np.percentile(prob_map,85)))
    binary = (prob_map>thresh).astype(np.uint8)
    print(f"Adaptive threshold (Otsu + percentile capped): {thresh:.4f}  "
          f"(pixels above threshold: {binary.sum()})")
    return binary, thresh


# =========================
# 11. Post-Processing
# =========================

def remove_small_and_elongated(binary: np.ndarray,
                                min_area: int,
                                max_aspect: float) -> np.ndarray:
    """
    Connected-component filter that removes:
        - Components smaller than min_area pixels  (noise, screenshot artifacts)
        - Components with width/height aspect ratio > max_aspect
          (vehicles, road markings, map UI elements)

    Parameters
    ----------
    binary     : uint8 binary mask (0 or 1)
    min_area   : minimum pixel area to keep
    max_aspect : maximum bounding-box aspect ratio to keep

    Returns
    -------
    cleaned binary mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)

    kept = removed_small = removed_shape = 0
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        h    = stats[lbl, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            removed_small += 1
            continue

        aspect     = (w / h)       if h > 0 else 0
        inv_aspect = (1.0 / aspect) if aspect > 0 else 0
        if aspect > max_aspect or inv_aspect > max_aspect:
            removed_shape += 1
            continue

        cleaned[labels == lbl] = 1
        kept += 1

    print(f"Component filter -- kept: {kept}, "
          f"removed (too small): {removed_small}, "
          f"removed (vehicle-like shape): {removed_shape}.")
    return cleaned


def morphological_smoothing(binary: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    1. Morphological close  -> fills small holes inside buildings
    2. Morphological open   -> removes thin protrusions / spurs
    3. Gaussian blur + re-threshold -> smooths jagged pixel-level edges
    """
    kernel   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened   = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  kernel, iterations=1)
    blurred  = cv2.GaussianBlur(opened.astype(np.float32), (3, 3), sigmaX=0.3)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed


# =========================
# 12. Watershed Instance Separation
# =========================

def watershed_separation(binary: np.ndarray) -> np.ndarray:
    """
    Splits merged/touching buildings using the distance-transform
    watershed algorithm.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, maskSize=5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg    = sure_fg.astype(np.uint8)

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg    = cv2.dilate(binary, kernel, iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 1] = 0

    img_3ch   = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers   = cv2.watershed(img_3ch, markers)
    separated = np.where(markers > 1, 1, 0).astype(np.uint8)

    before = int((binary > 0).sum())
    after  = int((separated > 0).sum())
    print(f"Watershed separation complete "
          f"(foreground pixels before={before}, after={after}).")
    return separated


# =========================
# 13. Polygonization
# =========================

def polygonize(binary: np.ndarray,
               epsilon: float,
               geotransform: tuple = None) -> list:
    """
    Converts the binary mask into simplified polygon footprints.

    Parameters
    ----------
    binary        : uint8 binary mask
    epsilon       : Douglas-Peucker simplification distance in pixels
    geotransform  : optional 6-element affine tuple from rasterio.
                    If None, coordinates stay in pixel space.

    Returns
    -------
    List of GeoJSON Feature dicts.
    """

    def _to_coords(pts_px):
        if geotransform is None:
            return pts_px
        ox, pw, _, oy, _, ph = geotransform
        return [[ox + pt[0] * pw, oy + pt[1] * ph] for pt in pts_px]

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    features = []
    if hierarchy is None:
        return features
    hierarchy = hierarchy[0]

    for idx, (contour, hier) in enumerate(zip(contours, hierarchy)):
        if hier[3] != -1:
            continue

        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue

        exterior_px = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
        exterior_px.append(exterior_px[0])
        exterior    = _to_coords(exterior_px)

        holes     = []
        child_idx = hier[2]
        while child_idx != -1:
            hole_approx = cv2.approxPolyDP(
                contours[child_idx], epsilon, closed=True
            )
            if len(hole_approx) >= 3:
                ring_px = [[int(pt[0][0]), int(pt[0][1])] for pt in hole_approx]
                ring_px.append(ring_px[0])
                holes.append(_to_coords(ring_px))
            child_idx = hierarchy[child_idx][0]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [exterior] + holes,
            },
            "properties": {
                "id":   idx,
                "area": int(cv2.contourArea(contour)),
                "crs":  "pixel" if geotransform is None else "geographic",
            },
        }
        features.append(feature)

    coord_type = "pixel" if geotransform is None else "map/geographic"
    print(f"Polygonization: {len(features)} building footprints "
          f"extracted ({coord_type} coordinates).")
    return features


def save_geojson(features: list, path: str) -> None:
    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"GeoJSON saved -> {path}")


# =========================
# 14. Full Pipeline
# =========================

def run_pipeline(image_path: str,
                 model_path: str = "IHUNet-100eps.keras",
                 is_screenshot: bool = True) -> dict:
    """
    End-to-end pipeline:
        large satellite image / web screenshot
            -> [screenshot preprocessing]  <- upscale, blur, normalize, CLAHE
            -> tile into 512x512 patches
            -> multi-scale (1.0x / 1.25x) + TTA (4 augmentations)
            -> probability map  <- Gaussian-blended tile stitching
            -> adaptive threshold (Otsu)
            -> remove small components + shape filter (vehicles / UI)
            -> morphological smoothing + boundary blur
            -> watershed instance separation
            -> polygonization  <- pixel or geo coordinates
            -> building footprints (GeoJSON + PNG mask + probability map)

    Parameters
    ----------
    image_path    : path to image file (.jpg / .png / .tif / .tiff)
    model_path    : path to trained Keras model
    is_screenshot : set True for web map screenshots (enables preprocessing)
                    set False for raw aerial / GeoTIFF imagery
    """

    # Load model
    model = load_model(model_path)

    # Load and normalize image
    print(f"\nLoading image: {image_path}")

    geotransform = None
    if HAS_RASTERIO and image_path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(image_path) as src:
            raw_arr      = src.read()
            raw          = np.transpose(raw_arr[:3], (1, 2, 0))
            t            = src.transform
            geotransform = (t.c, t.a, t.b, t.f, t.d, t.e)
        print(f"  GeoTIFF detected -- geotransform: {geotransform}")
    else:
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    image = raw.astype(np.float32) / 255.0
    print(f"  Image shape: {image.shape}")

    # Step 0: Screenshot preprocessing (domain adaptation)
    if is_screenshot:
        print("\n[0/6] Screenshot preprocessing (upscale + blur + normalize + CLAHE) ...")
        image = preprocess_screenshot(image, upscale=SCREENSHOT_UPSCALE)
        # edge sharpening
        blur = cv2.GaussianBlur(image, (0,0), 1.0)
        image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
        print(f"  Preprocessed image shape: {image.shape}")

    # Step 1: Tiled prediction (multi-scale + TTA)
    print("\n[1/6] Running tiled prediction (multi-scale + TTA) ...")
    prob_map = tile_predict(model, image)

    # Step 2: Adaptive threshold (Otsu)
    print("\n[2/6] Adaptive thresholding (Otsu) ...")
    binary, chosen_thresh = adaptive_threshold(prob_map)

    # Step 3: Remove small / elongated components
    print(f"\n[3/6] Filtering components "
          f"(min_area={MIN_AREA} px, max_aspect={MAX_ASPECT}) ...")
    binary = remove_small_and_elongated(binary, MIN_AREA, MAX_ASPECT)

    # Step 4: Morphological smoothing + boundary blur
    print(f"\n[4/6] Morphological smoothing + boundary blur "
          f"(kernel={MORPH_KERNEL}) ...")
    binary = morphological_smoothing(binary, MORPH_KERNEL)

    # Step 5: Watershed instance separation
    print("\n[5/6] Watershed instance separation ...")
    binary = watershed_separation(binary)

    # Step 6: Polygonization
    print("\n[6/6] Polygonizing footprints ...")
    features = polygonize(binary, APPROX_EPSILON, geotransform)

    # Save outputs
    cv2.imwrite(OUTPUT_MASK,     binary * 255)
    cv2.imwrite(OUTPUT_PROB_MAP, (prob_map * 255).astype(np.uint8))
    print(f"\nBinary mask saved     -> {OUTPUT_MASK}")
    print(f"Probability map saved -> {OUTPUT_PROB_MAP}")
    save_geojson(features, OUTPUT_GEOJSON)

    return {
        "image":        image,
        "prob_map":     prob_map,
        "binary":       binary,
        "features":     features,
        "threshold":    chosen_thresh,
        "geotransform": geotransform,
    }


# =========================
# 15. Visualization
# =========================

def show_results(results: dict) -> None:
    image    = results["image"]
    prob_map = results["prob_map"]
    binary   = results["binary"]
    features = results["features"]
    thresh   = results["threshold"]

    overlay = (image * 255).astype(np.uint8).copy()
    for feat in features:
        coords = feat["geometry"]["coordinates"][0]
        pts    = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(image)
    axes[0].set_title("Preprocessed Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Probability Map\n(multi-scale TTA)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title(f"Binary Mask\n(adaptive threshold threshold = {thresh:.3f})", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Building Footprints\n({len(features)} polygons)", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("pipeline_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Visualization saved -> pipeline_output.png")


# =========================
# 16. Entry Point
# =========================

if __name__ == "__main__":
    IMAGE_PATH    = "test image_04.png"     # Supports .jpg / .png / .tif / .tiff
    MODEL_PATH    = "IHUNet-100eps.keras"
    IS_SCREENSHOT = True                    # Set False for raw aerial / GeoTIFF input

    results = run_pipeline(IMAGE_PATH, MODEL_PATH, is_screenshot=IS_SCREENSHOT)
    show_results(results)