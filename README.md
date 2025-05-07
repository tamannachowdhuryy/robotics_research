Steps to Run the Model on Google Colab:

Clone the Repository:

bash
Copy
Edit
!git clone https://github.com/maelfabien/Multimodal-Emotion-Recognition.git
Upload Kaggle API Key:

Go to Kaggle.

Get your kaggle.json key from My Account.

Upload it to your Colab environment.

python
Copy
Edit
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
Download Required Datasets:

bash
Copy
Edit
!kaggle datasets download -d msambare/fer2013
!unzip fer2013.zip -d fer2013

!kaggle datasets download -d davilsena/ckdataset
!unzip ckdataset.zip -d CK+
Run the Training Script:

bash
Copy
Edit
!python3 final.py
Download the Trained Models to Your Computer:

After training, download the .pth model files to your local machine.

Move to VS Code:

Once the models are downloaded, move to VS Code.

Run:

bash
Copy
Edit
python3 app.py
Update the routes in app.py as needed.

