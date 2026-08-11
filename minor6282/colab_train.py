# Mental Disorder Detection – Google Colab Training Notebook
# ============================================================
# Run this notebook on Google Colab with GPU runtime enabled.
# Runtime → Change runtime type → T4 GPU
#
# STEP 0 – Mount Drive & Clone/Upload your repo
# ─────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# Option A: If you pushed to GitHub
# !git clone https://github.com/umleshyadav/MENTAL-DISORDER-DETECTION-BY-FACIAL-EXPRESSION.git /content/project
# %cd /content/project/minor6282

# Option B: If you uploaded the folder to Drive
# import shutil
# shutil.copytree('/content/drive/MyDrive/minor6282', '/content/project')
# %cd /content/project/minor6282

# ─────────────────────────────────────────────
# STEP 1 – Install dependencies
# ─────────────────────────────────────────────
# !pip install timm -q

# ─────────────────────────────────────────────
# STEP 2 – Upload dataset (if not in Drive)
# ─────────────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()   # upload datasets.zip then unzip below
# !unzip datasets.zip -d .

# ─────────────────────────────────────────────
# STEP 3 – Verify GPU
# ─────────────────────────────────────────────
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# ─────────────────────────────────────────────
# STEP 4 – Train ViT (emotion recognition)
# ─────────────────────────────────────────────
# %run train_vit.py

# ─────────────────────────────────────────────
# STEP 5 – Train LSTM (mental state prediction)
# ─────────────────────────────────────────────
# %run train_lstm.py

# ─────────────────────────────────────────────
# STEP 6 – Download trained weights
# ─────────────────────────────────────────────
# from google.colab import files
# files.download('best_vit.pth')
# files.download('lstm_model.pth')

print("All steps complete!")
