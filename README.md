# LC25000-Classifier 
Hybrid ensemble models combining PCA, MLP, and CNNs for classifying lung and colon histopathological images from the LC25000 dataset.

## 🗂️ Project Overview
This project implements hybrid machine learning and deep learning models to classify histopathological images from the LC25000 dataset, which contains images of lung (3 classes) and colon (2 classes) tissues. The study combines traditional feature reduction techniques (PCA) with classifiers like MLP and Deep CNNs, and finalizes predictions using ensemble voting methods.

## 🧩 Folder Structure
```bash
├── Dataset/
│   ├── colon_image_sets/
│   │   ├── colon_aca/         # Colon adenocarcinoma images
│   │   └── colon_n/           # Colon normal images
│   └── lung_image_sets/
│       ├── lung_aca/          # Lung adenocarcinoma images
│       ├── lung_n/            # Lung normal images
│       └── lung_scc/          # Lung squamous cell carcinoma images
├── models/
│   ├── pca_mlp_wrapper.py      # PCA + MLP wrapper class
│   ├── deep_cnn.py             # Deep CNN architecture code
│   ├── training_scripts.py     # Scripts for training models
│   └── ensemble_voting.py      # Ensemble voting implementation
├── notebooks/
│   └── Ensemble Classification of LC25000 Histopathological Lung and Colon Images.ipynb  # Main notebook 
├── requirements.txt            # Python dependencies
└── README.md                   # Project description and instructions
```
## 🚀 Methodology

### 1️⃣ Data Loading & Preprocessing
. Load and preprocess LC25000 histopathological images from Dataset/colon_image_sets and Dataset/lung_image_sets.

. Perform train-test split and encode target labels.

### 2️⃣ Model Development
. Train baseline classifiers: SVM, MLP, and simple CNN to benchmark initial accuracy.

### 3️⃣ Hybrid Models (PCA + Classifiers)
. Use PCA for dimensionality reduction of image features.

### 4️⃣ Train hybrid models like PCA + MLP and PCA + SVM.
. Transfer Learning Models
. Apply pretrained CNN architectures (e.g., EfficientNet, ResNet) adapted with fine-tuning on the dataset.

### 5️⃣ Select Top Models
. Evaluate models and select the top three (e.g., MLP, Deep CNN, PCA+MLP) based on performance metrics.

### 6️⃣ Ensemble Voting
. Create voting ensembles over the top models:
. Hard voting (majority vote)
. Soft voting (weighted probabilities)
. Evaluate both to choose the better ensemble.

### 7️⃣ Final Model Selection & Evaluation
. Identify best ensemble model combination (soft voting ensemble of MLP, Deep CNN, PCA+MLP)
. Compute metrics: Accuracy, Precision, Recall, F1-score, MAE, MSE, RMSE, Confusion Matrix, ROC & AUC
. Generate detailed visualizations for model performance.

## 🛠️ How to Run

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lc25000-histopath-classification.git
cd lc25000-histopath-classification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Open and run the Jupyter notebook:

## ⚙️ Dependencies
```bash
. Python 3.8+
. numpy, pandas
. scikit-learn
. tensorflow, keras, scikeras
. matplotlib, seaborn
```

## 🤝 Acknowledgments
. LC25000 Dataset contributors
. TensorFlow and Keras pretrained model repositories
. Open-source ML/DL community
