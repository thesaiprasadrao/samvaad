# Model Files Directory

This directory is intended for storing trained model files (.pt files).

## ⚠️ Important Note

The trained model files are **not included** in this repository because they exceed GitHub's file size limit (100MB per file).

## 📦 Getting Model Files

### Option 1: Train Your Own Models
Run the training scripts to generate your own model files:

```bash
# Train emotion and context models
python emotion_context_trainer.py

# Train language detection model
python language_detection_trainer.py

# Train pretrained models
python pretrained_model_trainer.py

# Main ALM training
python alm_model_trainer.py
```

### Option 2: Download from External Storage
If available, download the pre-trained models from:
- Google Drive
- Dropbox
- Other cloud storage

### Option 3: Use Git LFS
For large files, consider using Git Large File Storage (LFS):

```bash
git lfs track "*.pt"
git add .gitattributes
git add checkpoints/*.pt
git commit -m "Add model files with LFS"
git push origin main
```

## 📁 Expected Model Files

After training, you should have these files:
- `improved_emotion_model.pt` (~371MB)
- `simple_pretrained_context_model.pt` (~361MB)
- `simple_pretrained_emotion_model.pt` (~361MB)

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the models using the training scripts
4. Run the web interface: `python -m http.server 8000`

The models will be automatically loaded from this directory when available.
