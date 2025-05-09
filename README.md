# 🩻 X-Ray Anomaly Classifier

This project implements an end-to-end pipeline for anomaly classification in chest X-ray images, using the NIH ChestXray14 dataset.

## 🔧 Project Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/xray_anomaly.git
cd xray_anomaly
```

### 2. Set up the environment
```bash
pip install -r requirements.txt
```

### 3. Download the NIH ChestX-ray14 dataset

Place the images inside folders like:
```
data/xray_images/images_001/
data/xray_images/images_002/
...
```

(Within `notebooks/eda.ipynb`, there is a script to use
kagglehub to download locally)

Also make sure the metadata CSVs are in:
```
data/xray_images/BBox_List_2017.csv
data/xray_images/Data_Entry_2017.csv
```

### 4. Prepare the train/val splits

Ensure `train_val_list.txt` is located at:
```
data/xray_images/train_val_list.txt
```

Then run the split script:
```bash
python scripts/prepare_splits.py
```

This will generate:
- `train.txt`
- `val.txt`

---

## 🧠 Model Training

Train the model with:
```bash
python run.py
```

Outputs will be saved in:
```
saved_models/
outputs/
```

---

## 📊 Data Structure

- `BBox_List_2017.csv` — bounding boxes
- `Data_Entry_2017.csv` — patient metadata + labels
- `train.txt`, `val.txt` — filenames for splits

---

## 🧪 Components

| File/Folder                | Description                                 |
|----------------------------|---------------------------------------------|
| `src/dataset.py`           | Custom PyTorch dataset w/ bbox + labels     |
| `src/model.py`             | CNN model definition (ResNet/EfficientNet)  |
| `src/train.py`             | Training loop                               |
| `src/evaluate.py`          | Evaluation metrics + visualizations         |
| `src/gradcam.py`           | GradCAM attention heatmaps                  |
| `scripts/prepare_splits.py`| Generates train/val split                   |

---

## 🚀 Optional: Streamlit Demo

To launch a local web demo:
```bash
streamlit run app/app.py
```

---

## ✅ TODOs

- [x] Dataset loading with bbox and labels  
- [x] Dynamic folder indexing  
- [x] Train/val split script  
- [ ] Add training + evaluation  
- [ ] GradCAM visualization  
- [ ] Streamlit interactive demo  

---

## 📬 Contact

Questions or contributions welcome! Reach out at: `charleysanchez7@gmail.com`