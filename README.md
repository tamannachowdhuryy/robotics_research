
# Emotion Detection Model Setup 🚀

This guide provides the steps to set up, train, and deploy emotion detection models using Google Colab and VS Code.

## 📁 Setup in Google Colab

### 1. Clone the Repository

```bash
!git clone https://github.com/maelfabien/Multimodal-Emotion-Recognition.git
```

### 2. Upload Kaggle API Key

- Get your `kaggle.json` key from [Kaggle](https://www.kaggle.com/).  
- Upload it to your Colab environment.
- Everything is there just incase its not :)

```python
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
```

### 3. Download Required Datasets

```bash
!kaggle datasets download -d msambare/fer2013
!unzip fer2013.zip -d fer2013

!kaggle datasets download -d davilsena/ckdataset
!unzip ckdataset.zip -d CK+
```

### 4. Run the Training Script

```bash
!python3 final.py
```

## 💻 Move to VS Code

Once the models are downloaded, move to VS Code and run:

```bash
python3 app.py
```

- Make sure to adjust the routes in `app.py` if needed.

---

🎉 That's it! You’re all set to start detecting emotions with your custom-trained models! 😎
