# Malaria_detection_using_pretrained_deep_learning_models

## Overview

Malaria Detection Portal is a Streamlit-based web application that leverages deep learning models to classify blood smear images as **Parasitized** or **Uninfected**. The app utilizes pre-trained convolutional neural networks (MobileNetV2 and InceptionV3) for accurate prediction and provides Grad-CAM visualizations to explain model decisions, enhancing interpretability and trust.

> **Note:** This tool is intended for **educational and research purposes only** and should not be used as a medical diagnostic tool.

---

## Features

- Upload blood smear images in JPG, JPEG, or PNG format.
- Select between MobileNetV2 and InceptionV3 deep learning models.
- Real-time prediction of malaria infection status.
- Confidence score with interpretative guidance on prediction reliability.
- Grad-CAM heatmap visualization highlighting image regions influencing the model's decision.
- Informative UI designed for accessibility and user engagement.

---
## Tech Stack
- Python 3.10+
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, scikit-learn
- Streamlit (for app interface)

---
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/malaria-detection-ai.git](https://github.com/Ansariricky/Malaria_detection_using_pretrained_deep_learning_models.git)
   cd malaria-multi-model-app

2.Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```
3.Install dependencies:
   ```bash
   pip install -r requirements.txt
```
4.Launch the app:
   ```bash
   streamlit run app.py
```

---
## Demo
<img width="1886" height="887" alt="Screenshot 2025-08-04 005957" src="https://github.com/user-attachments/assets/fbd65085-cbd8-43c1-8a4b-5d2f47fc4f0a" />

Here the blood smear images is uploaded and results are calculated based on the model chosen and image data.

<img width="1877" height="855" alt="Screenshot 2025-08-04 010048" src="https://github.com/user-attachments/assets/21d19fff-81d6-441d-8bb1-b76b62de991d" />

---
## Folder Structure

```bash
malaria-multi-model-app/
├── .gitignore
├── app.py
├── folder-structure.txt
├── README.md
├── requirements.txt
└── models/
├── InceptionV3.h5
└── MobileNetV2.h5

```

---
## Authors
   This project was developed as part of a research internship under the guidance of Jadavpur University.
   * Rekibuddin Ansari – [LinkedIn](www.linkedin.com/in/rekibuddin-ansari-447772279) | [GitHub](https://github.com/Ansariricky)
---

## Future Work
   * Deploy to the Cloud(e.g. Streamlit Cloud/ HuggingFace Spaces)
   * Expand to multi-class classification (other blood-related diseases)
   
---
