
# MNIST Digit Classifier – FastAPI + React + AWS

This project is a full-stack AI application that allows users to draw a digit in the browser and receive model predictions with confidence scores.

It combines a Python backend for inference with a modern React frontend, and is deployed on AWS.

---

## 🚀 Project Overview

* Users draw a digit (0–9) in a web interface
* The image is sent to a backend API
* A trained PyTorch model performs inference
* The system returns:

  * The most probable digit
  * The top-3 predictions with confidence scores

---

## 🧠 Model

* Dataset: MNIST
* Model: CNN
* Input: 28x28 grayscale image
* Output: Probability distribution over 10 classes (digits 0–9)

---

## 🏗️ Architecture

Frontend (React + Vite)
→ Sends image via HTTP request
Backend (FastAPI)
→ Loads trained model (`.pt`)
→ Computes probabilities (softmax)
→ Returns top predictions

---

## 📂 Project Structure

```
.
├── backend/
│   ├── main.py
│   ├── model/
│   │   └── mnist_model.pt
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── index.html
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

### 2. Backend setup

```
cd backend
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Run backend:

```
uvicorn main:app --reload
```

Backend will run on:

```
http://127.0.0.1:8000
```

---

### 3. Frontend setup

```
cd frontend
npm install
npm run dev
```

Frontend will run on:

```
http://localhost:5173
```

---

## 🌐 API Endpoint

### POST `/predict`

**Output:**

```
{
  "prediction": 7,
  "confidence": 0.98,
  "top_3": [
    {"digit": 7, "confidence": 0.98},
    {"digit": 9, "confidence": 0.01},
    {"digit": 1, "confidence": 0.01}
  ]
}
```

---

## ☁️ Deployment

The backend is deployed on AWS using a load balancer.

Example endpoint:

```
http://your-load-balancer-url/predict
```

Frontend communicates with the backend using:

```
VITE_API_BASE_URL
```

---

## 🧪 Demo

* Draw a digit in the canvas
* Click predict
* View:

  * Main prediction
  * Top-3 probabilities

---

## 📌 Future Improvements

* Support mobile interaction
* Improve UI/UX

---

## 👨‍💻 Author

Carlos Mario Quiroga
Electronic Engineer & MSc in AI


