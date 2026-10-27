# 🤟 ISL Alphabet Interpreter Using Python and Machine Learning

A real-time **Indian Sign Language (ISL) alphabet recognition system** developed using Python, OpenCV, MediaPipe, and a Random Forest classifier. 📸 The application captures hand gestures through a webcam, extracts hand landmarks, predicts the corresponding alphabet, and displays the recognized text.

## ✨ Features

- 🤚 Real-time hand detection using MediaPipe.
- 📍 Extraction and normalization of 3D hand landmark coordinates.
- 🌳 Random Forest classification for recognizing English alphabet labels (A–Z).
- ⏳ Temporal stabilization to reduce inconsistent predictions.
- 🎯 Confidence-based prediction filtering.
- 💬 Display of the predicted alphabet and accumulated recognized text.

## 🛠️ Technologies Used

- 🐍 **Python** – Application development and model training.
- 📷 **OpenCV** – Webcam access and video display.
- 🤚 **MediaPipe** – Hand landmark detection.
- 🔢 **NumPy** – Numerical operations and feature preparation.
- 🌳 **scikit-learn** – Random Forest classification.
- 💾 **Joblib** – Saving and loading the trained model.

## 📂 Project Structure

```text
ISL/
├── isl_interpreter.py
├── prepare_dataset.py
├── train_model.py
├── requirements.txt
├── models/
│   └── isl_alphabet_rf.joblib
└── README.md
```

The dataset directory and any generated intermediate files are not shown above. Their exact locations depend on the dataset preparation setup.

## 💻 System Requirements

- 🪟 Windows 10/11 with WSL2 enabled, or a compatible Linux system.
- 🐧 Ubuntu installed under WSL2 for Windows users.
- 🐍 Python 3 and `pip`.
- 📸 A working webcam.
- 🔌 USB/IP webcam passthrough configured for WSL2 when using the Windows integrated camera.
- 📁 The project files and trained model.
- 🌐 Internet access for installing dependencies.

> ⚠️ **Important:** Windows users should run the Python application inside the WSL2 Ubuntu terminal, not directly in Windows Command Prompt. The webcam must be attached to WSL2 before running the application.

## 🚀 Installation and Setup

### 1. 🐧 Set Up WSL2 and Ubuntu

If WSL2 and Ubuntu are already installed, open the Ubuntu terminal and continue to Step 2.

Otherwise, open PowerShell as Administrator and install WSL using:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if prompted and complete the Ubuntu initial setup.

### 2. 📸 Connect the Webcam to WSL2

Windows users need to pass the webcam through to WSL2 using `usbipd-win`.

Install `usbipd-win` if it is not already installed. Open PowerShell as Administrator and list the connected USB devices:

```powershell
usbipd list
```

Identify the webcam's BUSID. In this project's setup, the webcam used BUSID `5-1`. Your BUSID may differ.

**Bind the device:**

```powershell
usbipd bind --busid 5-1
```

**Attach it to WSL2:**

```powershell
usbipd attach --wsl --busid 5-1
```

If the BUSID changes, replace `5-1` with the value shown by `usbipd list`. Device binding may require administrator privileges.

Keep the Ubuntu terminal open for the next steps. If the webcam is disconnected after a WSL restart, attach it again using the command above.

### 3. 📁 Open the Project Directory

Inside Ubuntu, navigate to the project directory.

For example, if the project is stored in `C:\Users\[user]\Desktop\ISL`:

```bash
cd /mnt/c/Users/[user]/Desktop/ISL
```

Replace the path if the project is stored elsewhere.

### 4. 🐍 Create and Activate the Virtual Environment

Install `virtualenv` if necessary:

```bash
python3 -m pip install --user virtualenv
```

Create the virtual environment:

```bash
virtualenv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, the terminal should display `(venv)` at the beginning of the prompt.

> 💡 If your Ubuntu installation does not have Python or pip installed, install them using `sudo apt update` followed by `sudo apt install python3 python3-pip python3-venv`.

### 5. 📦 Install Project Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs the Python packages listed in `requirements.txt`.

If installation fails because of a missing system library or an incompatible Python version, check the error message and install the required system package or compatible dependency version.

### 6. 🔍 Verify Webcam Access

Before launching the interpreter, check that WSL2 can access the webcam:

```bash
ls /dev/video*
```

If the device is attached successfully, one or more video device entries should appear.

You can also test the webcam from Python using OpenCV:

```bash
python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_V4L2); print('Camera available:', cap.isOpened()); cap.release()"
```

If the camera is unavailable, verify the USB/IP attachment and device permissions before proceeding.

### 7. ▶️ Run the ISL Interpreter

Ensure the virtual environment is active, the webcam is attached, and the trained model file exists at the path expected by the application.

Launch the application:

```bash
python isl_interpreter.py
```

🎉 The application opens a webcam window, detects hand landmarks, predicts the alphabet, and displays the recognized text.

## ⌨️ Application Controls

| Key | Function |
|---|---|
| `B` | ⬅️ Delete the last recognized character |
| `Space` | ␣ Insert a space |
| `X` | 🧹 Clear the recognized text |
| `Q` | 🚪 Quit the application |

The application uses confidence filtering and temporal stabilization to reduce accidental character recognition. Hold a gesture steady to allow the prediction to stabilize.

## 🧠 Dataset Preparation and Model Training

The project also includes `prepare_dataset.py` and `train_model.py` for preparing the dataset and training the classifier.

These scripts are intended for dataset preparation and model training rather than normal application startup.

If you need to regenerate the trained model:

1. 📥 Download and extract the [RealSign Indian Sign Language Alphabet Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset).
2. 📂 Place the extracted dataset in the location expected by `prepare_dataset.py`.
3. 🔧 Run the dataset preparation script:

   ```bash
   python prepare_dataset.py
   ```

4. 🌳 Run the training script:

   ```bash
   python train_model.py
   ```

5. 💾 Confirm that the generated model is saved to the location expected by `isl_interpreter.py`, typically `models/isl_alphabet_rf.joblib`.

> **Note:** The exact dataset directory and intermediate output paths must match those configured in the Python scripts. If the trained model is already included with the project, you can skip dataset preparation and training and launch the interpreter directly.

## 🛠️ Troubleshooting

### 📸 Webcam Not Detected

- Run `usbipd list` in Windows PowerShell.
- Attach the webcam to WSL2 using `usbipd attach --wsl --busid <BUSID>`.
- Check for `/dev/video*` inside Ubuntu.

### 📦 Module Not Found

- Activate the virtual environment and run `pip install -r requirements.txt` again.

### 💾 Trained Model Not Found

- Verify that the model file exists at the path configured in `isl_interpreter.py`.
- Run the dataset preparation and training scripts if the model needs to be generated.

### 🎯 Unstable Predictions

- Ensure the hand is clearly visible and adequately lit.
- Hold the gesture steady and keep the hand within the camera frame.

### 🔌 Camera Works in Windows but Not in Ubuntu

The Windows camera being functional does not automatically make it accessible to WSL2. Confirm that USB/IP passthrough is configured and that the device is attached.

## ⚠️ Limitations

- The system recognizes static alphabet gestures rather than complete sign language sentences.
- Recognition quality depends on lighting, camera quality, hand visibility, and gesture execution.
- The application is designed for the supported alphabet classes and is not a complete ISL translation system.

## 📚 Dataset

- [RealSign – Indian Sign Language Alphabet Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)

---

<div align="center">

### 🤟 Making Sign Language Recognition More Accessible

**Built with 🐍 Python and ❤️ for machine learning**

</div>
