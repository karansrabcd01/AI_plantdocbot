# Model Training Notebooks

This directory contains Jupyter notebooks for training the PlantDocBot machine learning models.

## 📓 Notebooks

### 1. ImageClassification.ipynb
**Purpose:** Train the CNN model for image-based plant disease detection

**Contents:**
- Data loading and preprocessing
- CNN architecture definition
- Model training and validation
- Performance evaluation
- Model export to `.pth` format

**Dataset:** PlantVillage dataset (38 classes)

**Output:** `ImageClassification_model_weights.pth`

---

### 2. TextClassifier.ipynb
**Purpose:** Train the BERT-based model for text-based disease diagnosis

**Contents:**
- Text data preparation
- BERT model fine-tuning
- Label encoding
- Training and evaluation
- Model export to HuggingFace format

**Dataset:** Plant disease symptom descriptions

**Output:** `text_classifier_model/` directory with all model files

---

## 🚀 Running the Notebooks

### Prerequisites

```bash
pip install jupyter notebook
pip install torch torchvision transformers
pip install pandas numpy matplotlib scikit-learn
```

### Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

### Training Steps

1. **Download Dataset** (if needed)
   - PlantVillage dataset for images
   - Symptom descriptions for text

2. **Open Notebook**
   - Click on the desired notebook

3. **Run Cells**
   - Execute cells sequentially
   - Monitor training progress

4. **Save Models**
   - Models will be saved to `Backend/models/`

---

## 📊 Model Performance

### Image Classification Model
- **Architecture:** Custom CNN
- **Accuracy:** ~95% on test set
- **Classes:** 38 plant diseases
- **Input:** 224x224 RGB images

### Text Classification Model
- **Architecture:** BERT-based
- **Accuracy:** ~92% on test set
- **Classes:** 38 plant diseases
- **Input:** Text descriptions

---

## 💡 Tips

- **GPU Recommended:** Training is much faster with GPU
- **Batch Size:** Adjust based on available memory
- **Epochs:** More epochs = better accuracy (watch for overfitting)
- **Data Augmentation:** Improves model generalization

---

## 🔧 Troubleshooting

**Out of Memory:**
- Reduce batch size
- Use smaller model
- Clear GPU cache

**Poor Accuracy:**
- Increase epochs
- Add data augmentation
- Tune hyperparameters

**Slow Training:**
- Use GPU instead of CPU
- Reduce dataset size for testing
- Use mixed precision training

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

**Happy Training! 🎓**
