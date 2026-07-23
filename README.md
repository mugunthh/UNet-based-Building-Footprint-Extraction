# 🏢 UNet-based Building Footprint Extraction

A deep learning project for automatic building footprint extraction from high-resolution satellite imagery using a U-Net-based semantic segmentation model.

## 📌 Overview

Building footprint extraction is a semantic segmentation task where each pixel in a satellite image is classified as either **building** or **background**. This project implements a U-Net architecture trained to accurately identify building regions and generate vector building footprints.

The workflow includes image preprocessing, model inference, mask generation, post-processing, and conversion of predicted masks into GeoJSON building polygons.

---

## 🚀 Features

- U-Net based semantic segmentation
- Automatic building footprint extraction
- High-resolution satellite image support
- Binary building mask generation
- Polygon extraction from segmentation masks
- GeoJSON output for GIS applications
- Visualization of extracted footprints

---

## 📂 Project Structure

```
UNet-based-Building-Footprint-Extraction/
│
├── assets/
│   ├── building_footprints.geojson
│   ├── building_footprints_02.geojson
│   └── pipeline_output_23.png
│
├── code/
│   └── BUILDING FOOTPRINT EXTRACTION.py
│
├── model/
│   └── IHUNet-100eps.keras
│
├── LICENSE
└── README.md
```

---

## 🧠 Model

This project uses a trained **U-Net** model for binary semantic segmentation.

**Architecture**

- Encoder
- Bottleneck
- Decoder
- Skip Connections
- Sigmoid Output Layer

The trained model is provided as:

```
model/IHUNet-100eps.keras
```

---

## ⚙️ Requirements

Install the required Python libraries:

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install geopandas
pip install rasterio
pip install shapely
pip install scikit-image
```

or

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the extraction pipeline:

```bash
python "code/BUILDING FOOTPRINT EXTRACTION.py"
```

The script performs:

1. Load satellite image
2. Preprocess image
3. Load trained U-Net model
4. Predict building mask
5. Apply post-processing
6. Extract polygons
7. Export GeoJSON
8. Visualize results

---

## 📤 Output

The project generates:

### Building Masks

Binary segmentation masks showing detected buildings.

### GeoJSON Files

```
assets/building_footprints.geojson
assets/building_footprints_02.geojson
```

These files can be opened in:

- QGIS
- ArcGIS Pro
- GeoPandas
- Leaflet
- OpenLayers

---

## 📸 Sample Output

Example prediction:

```
assets/pipeline_output_23.png
```

The output image illustrates:

- Original satellite image
- Predicted building mask
- Extracted building footprints

---

## 🛰️ Applications

- Urban planning
- Smart cities
- GIS mapping
- Disaster management
- Infrastructure monitoring
- Land use analysis
- Remote sensing
- Change detection

---

## 📖 Workflow

```
Satellite Image
        │
        ▼
 Image Preprocessing
        │
        ▼
     U-Net Model
        │
        ▼
 Binary Segmentation Mask
        │
        ▼
 Morphological Processing
        │
        ▼
 Contour Extraction
        │
        ▼
 Polygon Generation
        │
        ▼
 GeoJSON Output
```

---

## 📊 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Rasterio
- GeoPandas
- Shapely
- Scikit-image

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mugunth Sanjay P.**

B.E. Geoinformatics  
College of Engineering, Guindy (Anna University)

GitHub: https://github.com/<your-github-username>

---

## ⭐ Future Improvements

- Support for large satellite mosaics
- Batch image processing
- Multi-class segmentation
- Building height estimation
- Web-based inference application
- Docker deployment
- ONNX model export
- Cloud deployment using Google Earth Engine or Azure
