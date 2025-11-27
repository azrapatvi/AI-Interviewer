# AI Interviewer 🚀

A **real-time AI-powered interview platform** built with **FastAPI**, **DeepFace**, and **Google Gemini API**. This system allows users to register, verify their identity via face recognition, participate in interviews, receive AI-generated feedback, and download detailed performance reports in PDF.

---

## Features ✨

- **User Authentication:** Secure login and registration with face verification.
- **Face Recognition:** Live face verification using DeepFace and OpenCV.
- **AI Interview Questions:** Generate domain-specific interview questions using Gemini API.
- **Real-Time Monitoring:** Detect multiple faces and monitor authorized user presence during interviews.
- **Speech Feedback:** Text-to-speech feedback using pyttsx3.
- **Audio Transcription:** Upload and transcribe audio using Google Generative AI.
- **Performance Dashboard:** Track interview metrics: confidence, accuracy, correctness, and overall score.
- **PDF Reports:** Download detailed interview reports with AI-generated analysis.
- **MongoDB Integration:** Stores user profiles, interview questions, answers, and performance data.

---

## Tech Stack 🛠️

- **Backend:** FastAPI, Python
- **AI & ML:** DeepFace, Google Gemini API
- **Database:** MongoDB Atlas
- **Frontend:** Jinja2 Templates, HTML/CSS/JS
- **Audio & Speech:** pyttsx3, pydub, Google Generative AI
- **PDF Reports:** ReportLab
- **Video:** OpenCV for webcam integration

---

## Install dependencies:
```
pip install -r requirements.txt
```

## Usage 🎯

Register: Sign up with your email, password, and face capture.
Login: Authenticate and verify your face.
Start Interview: Choose a domain and answer AI-generated questions
Receive Feedback: Get spoken and written AI feedback on your answers.
View Reports: Check performance dashboard and download PDF reports.


## structure
```
AI-Interviewer/
├── main.py                # FastAPI application
├── templates/             # HTML templates
├── static/                # CSS, JS, images
├── requirements.txt       # Python dependencies
└── README.md

```

