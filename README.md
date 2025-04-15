# 🧠 IT-JIM Task 1 - DL

A PyTorch-based project for training and testing a binary classifier on image data. Includes CLI support for training and inference.

---

## 📁 Project Structure

```
it-jim_ak1_2025/
├── data/
│   ├── train/          # Training images
│   └── test/           # Test images
├── models/             # Trained models (saved here)
├── src/                # Source code modules
├── main.py             # CLI for training/testing
├── requirements.txt    # Dependencies
├── README.md           # This file
```

---

## ⚖️ Setup Instructions

0. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd it-jim_ak1_2025
   ```

1. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify folder structure**
   Ensure your training and test images are in:
   - `data/train/`
   - `data/test/`

   Image filenames should follow the format: `image_<id>_<label>.png`

---

## ▶️ How to Use

### Train the Model
```bash
python main.py train
```

**Optional Arguments:**
```bash
--data-dir   Path to training data (default: ./data/train)
--epochs     Number of training epochs (default: from config)
```

Example:
```bash
python main.py train --data-dir ./data/trainee_dataset/train --epochs 20
```

---

### Run Inference on Test Set
```bash
python main.py test
```

**Optional Arguments:**
```bash
--data-dir   Path to test data (default: ./data/test)
--output     Output CSV filename (default: test_predictions.csv)
```

Example:
```bash
python main.py test --data-dir ./data/trainee_dataset/test --output results.csv
```

---

## 📊 Evaluation Metrics
- **Micro F1 Score** (on full test set)
- **Artifact-only Accuracy** (optional)

---

## 🚧 Notes
- The best model is saved to `models/best_model.pth`


---

## 🚀 Author
Made with ❤️ by Arseny K