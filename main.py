from fastapi import FastAPI, Request, Form, Query, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pymongo import MongoClient
import numpy as np
import threading
import requests
import os
import cv2
import pyttsx3
import atexit
import time
from deepface import DeepFace
import speech_recognition as sr
from datetime import datetime
import google.generativeai as genai
from bson import ObjectId
from pydub import AudioSegment
import tempfile
from pydub.utils import which
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io
import traceback
import base64

# pydub ffmpeg
AudioSegment.converter = which("ffmpeg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------------------- Configuration -------------------- #
GEMINI_API_KEY = "AIzaSyB01_ahSqRc-N2uPr4SoXF1BfXNEL1G9J0"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- FastAPI Initialization -------------------- #
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-very-secret-key")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- MongoDB Connection -------------------- #
client = MongoClient("mongodb+srv://interviewerai12_db_user:ai12@aiinterviewer.kg5hisp.mongodb.net/")
mydb = client["interviewer"]
mycollection = mydb["registered"]
collection = mydb["questions"]
interviews_collection = mydb["interviews"]

# -------------------- Global Variables -------------------- #
camera = cv2.VideoCapture(0)
interview_data = {
    "questions": [],
    "answers": [],
    "feedback": [],
    "emotions": [],
    "current_question_index": 0,
    "interview_id": None,
    "domain": None,
    "user_id": None
}
face_verified = False 

def release_camera():
    if camera.isOpened():
        camera.release()

atexit.register(release_camera)

FACE_IMAGES_DIR = "static/face_images"

if not os.path.exists(FACE_IMAGES_DIR):
    os.makedirs(FACE_IMAGES_DIR)

# -------------------- Face Recognition Functions -------------------- #
def save_face_image(user_id, image_data):
    """Save user's face image for future verification"""
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        face_path = os.path.join(FACE_IMAGES_DIR, f"{user_id}.jpg")
        cv2.imwrite(face_path, img)
        
        return face_path
    except Exception as e:
        print(f"❌ Error saving face image: {e}")
        return None

def verify_face(user_id, current_image_data):
    """Verify if current face matches registered face"""
    try:
        stored_face_path = os.path.join(FACE_IMAGES_DIR, f"{user_id}.jpg")
        
        if not os.path.exists(stored_face_path):
            return False, "No registered face found for this user"
        
        image_bytes = base64.b64decode(current_image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        current_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        temp_path = os.path.join(FACE_IMAGES_DIR, f"temp_{user_id}.jpg")
        cv2.imwrite(temp_path, current_img)
        
        result = DeepFace.verify(
            img1_path=stored_face_path,
            img2_path=temp_path,
            model_name='VGG-Face',
            enforce_detection=True
        )
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result['verified']:
            return True, "Face verified successfully"
        else:
            return False, f"Face does not match. Distance: {result['distance']:.2f}"
            
    except Exception as e:
        print(f"❌ Face verification error: {e}")
        return False, f"Verification failed: {str(e)}"

# -------------------- Speech Control -------------------- #
engine_lock = threading.Lock()
engine_instance = None

def stop_speaking():
    global engine_instance
    with engine_lock:
        if engine_instance is not None:
            try:
                engine_instance.stop()
                engine_instance = None
                print("🛑 Speech stopped.")
            except Exception as e:
                print("Error stopping speech:", e)

@app.post("/stop_speaking")
async def stop_speaking_api():
    stop_speaking()
    return {"success": True, "message": "Speech stopped"}

# -------------------- Basic Routes -------------------- #
@app.get("/", response_class=HTMLResponse)
async def login(request: Request, message: str = Query(default=""), flag: str = Query(default="")):
    return templates.TemplateResponse("login.html", {"request": request, "message": message, "flag": flag})

@app.post("/")
async def login_sub(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        user = mycollection.find_one({"email": email})
        if not user:
            return templates.TemplateResponse("login.html", {"request": request, "message": "Email not registered!", "flag": "danger"})
        if user.get("password") != password:
            return templates.TemplateResponse("login.html", {"request": request, "message": "Incorrect password!", "flag": "danger"})
        
        request.session["temp_user_id"] = str(user["_id"])
        request.session["temp_user_name"] = user.get("name", "")
        request.session["temp_email"] = user.get("email", "")
        
        return RedirectResponse(url="/verify_face", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("login.html", {"request": request, "message": f"Something went wrong: {e}", "flag": "danger"})

@app.get("/register", response_class=HTMLResponse)
async def register(request: Request, message: str = Query(default=""), flag: str = Query(default="")):
    return templates.TemplateResponse("register.html", {"request": request, "message": message, "flag": flag})

@app.post("/register")
async def submit(request: Request):
    try:
        form_data = await request.form()
        name = form_data.get("name")
        email = form_data.get("email")
        password = form_data.get("password")
        confirmpassword = form_data.get("confirmpassword")
        face_image = form_data.get("face_image")
        
        if password != confirmpassword:
            return templates.TemplateResponse("register.html", {"request": request, "message": "Passwords do not match!", "flag": "danger"})
        if mycollection.find_one({"email": email}):
            return templates.TemplateResponse("register.html", {"request": request, "message": "Email already registered!", "flag": "danger"})
        
        if not face_image:
            return templates.TemplateResponse("register.html", {"request": request, "message": "Please capture your face image!", "flag": "danger"})
        
        result = mycollection.insert_one({"name": name, "email": email, "password": password})
        user_id = str(result.inserted_id)
        
        face_path = save_face_image(user_id, face_image)
        
        if not face_path:
            mycollection.delete_one({"_id": result.inserted_id})
            return templates.TemplateResponse("register.html", {"request": request, "message": "Failed to save face image!", "flag": "danger"})
        
        mycollection.update_one(
            {"_id": result.inserted_id},
            {"$set": {"face_image_path": face_path}}
        )
        
        request.session["user_id"] = user_id
        request.session["user_name"] = name
        request.session["email"] = email
        
        return RedirectResponse(url="/index", status_code=303)
    except Exception as e:
        return templates.TemplateResponse("register.html", {"request": request, "message": f"Something went wrong: {e}", "flag": "danger"})

# -------------------- Face Verification Route -------------------- #
@app.get("/verify_face", response_class=HTMLResponse)
async def verify_face_page(request: Request):
    temp_user_id = request.session.get("temp_user_id")
    if not temp_user_id:
        return RedirectResponse(url="/", status_code=303)
    
    return templates.TemplateResponse("verify_face.html", {"request": request})

@app.post("/verify_face")
async def verify_face_submit(request: Request):
    try:
        form_data = await request.form()
        face_image = form_data.get("face_image")
        temp_user_id = request.session.get("temp_user_id")
        
        if not temp_user_id:
            return JSONResponse({"success": False, "message": "Session expired"})
        
        if not face_image:
            return JSONResponse({"success": False, "message": "No face image provided"})
        
        verified, message = verify_face(temp_user_id, face_image)
        
        if verified:
            request.session["user_id"] = request.session.pop("temp_user_id")
            request.session["user_name"] = request.session.pop("temp_user_name")
            request.session["email"] = request.session.pop("temp_email")
            
            return JSONResponse({"success": True, "message": "Face verified! Redirecting...", "redirect": "/index"})
        else:
            return JSONResponse({"success": False, "message": message})
            
    except Exception as e:
        print(f"❌ Face verification error: {e}")
        return JSONResponse({"success": False, "message": f"Verification failed: {str(e)}"})

# -------------------- Enhanced Face Recognition Functions -------------------- #
def count_faces_in_image(image_data):
    """Count number of faces in the image (LOW sensitivity version)"""
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✨ Reduced sensitivity settings
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,   # increased from 1.1 → reduces false faces
            minNeighbors=7,     # increased from 5 → requires stronger face evidence
            minSize=(120, 120)  # bigger face required → stops detecting background
        )

        return len(faces), img
    except Exception as e:
        print(f"❌ Error counting faces: {e}")
        return -1, None


def verify_face_enhanced(user_id, current_image_data):
    """Enhanced face verification with multiple face detection"""
    try:
        face_count, current_img = count_faces_in_image(current_image_data)
        
        if face_count == -1:
            return False, "Error processing image"
        
        if face_count == 0:
            return False, "No face detected. Please ensure your face is visible."
        
        if face_count > 1:
            return False, f"Multiple faces detected ({face_count} people). Only the registered candidate should be visible."
        
        stored_face_path = os.path.join(FACE_IMAGES_DIR, f"{user_id}.jpg")
        
        if not os.path.exists(stored_face_path):
            return False, "No registered face found for this user"
        
        temp_path = os.path.join(FACE_IMAGES_DIR, f"temp_{user_id}.jpg")
        cv2.imwrite(temp_path, current_img)
        
        try:
            result = DeepFace.verify(
                img1_path=stored_face_path,
                img2_path=temp_path,
                model_name='VGG-Face',
                enforce_detection=True
            )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result['verified']:
                return True, "Face verified successfully"
            else:
                return False, f"Face does not match registered user. Confidence: {(1 - result['distance']) * 100:.1f}%"
        
        except Exception as verify_error:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"DeepFace verification error: {verify_error}")
            return False, "Face verification failed. Please ensure good lighting and face the camera directly."
            
    except Exception as e:
        print(f"❌ Face verification error: {e}")
        return False, f"Verification failed: {str(e)}"

# -------------------- Interview Face Check -------------------- #
@app.post("/check_interview_face")
async def check_interview_face(request: Request):
    """Check if the correct person is giving the interview with multiple face detection"""
    try:
        form_data = await request.form()
        face_image = form_data.get("face_image")
        user_id = request.session.get("user_id")
        
        if not user_id:
            return JSONResponse({"success": False, "message": "Not logged in"})
        
        if not face_image:
            return JSONResponse({"success": False, "message": "No face detected. Please ensure your camera is working."})
        
        verified, message = verify_face_enhanced(user_id, face_image)
        
        global face_verified
        face_verified = verified
        
        return JSONResponse({"success": verified, "message": message})
        
    except Exception as e:
        print(f"❌ Interview face check error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "message": f"System error: {str(e)}"})

# -------------------- Continuous monitoring endpoint -------------------- #
@app.post("/monitor_interview_face")
async def monitor_interview_face(request: Request):
    """Continuous monitoring during interview"""
    try:
        form_data = await request.form()
        face_image = form_data.get("face_image")
        user_id = request.session.get("user_id")
        
        if not user_id:
            return JSONResponse({
                "success": False, 
                "message": "Session expired",
                "action": "terminate"
            })
        
        if not face_image:
            return JSONResponse({
                "success": False, 
                "message": "No face detected",
                "action": "warning"
            })
        
        verified, message = verify_face_enhanced(user_id, face_image)
        
        if not verified:
            if "Multiple faces" in message:
                return JSONResponse({
                    "success": False,
                    "message": message,
                    "action": "multiple_faces"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": message,
                    "action": "warning"
                })
        
        return JSONResponse({
            "success": True,
            "message": "Authorized user verified",
            "action": "continue"
        })
        
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        return JSONResponse({
            "success": False,
            "message": str(e),
            "action": "warning"
        })

# -------------------- REAL-TIME CONFIDENCE ANALYSIS -------------------- #
@app.post("/analyze_confidence")
async def analyze_confidence(request: Request):
    try:
        form = await request.form()
        img_base64 = form.get("face_image")

        if not img_base64:
            return JSONResponse({"success": False, "message": "No image"})

        # Decode base64
        img_data = base64.b64decode(img_base64.split(',')[1])
        img_np = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"success": False, "message": "Decode error"})

        # DeepFace emotion analysis
        try:
            analysis = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=True
            )
            if isinstance(analysis, list):
                analysis = analysis[0]

            emotions = analysis.get("emotion", {})
        except Exception as e:
            emotions = {}

        # If DeepFace failed, fallback to brightness
        if not emotions or sum(emotions.values()) == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            fallback_conf = int((brightness / 255) * 100)
            return JSONResponse({
                "success": True,
                "confidence": fallback_conf,
                "emotion": "unknown",
                "method": "fallback"
            })

        # Normalize emotion scores to percentages
        total = sum(emotions.values())
        emotions_percent = {k: (v / total) * 100 for k, v in emotions.items()}

        # Confidence formula: happy + neutral positively, negative emotions reduce
        confidence = (
            emotions_percent.get("happy", 0) * 1.2 +
            emotions_percent.get("neutral", 0) * 1.0 -
            emotions_percent.get("sad", 0) * 0.8 -
            emotions_percent.get("fear", 0) * 0.7 -
            emotions_percent.get("angry", 0) * 0.6 +
            emotions_percent.get("surprise", 0) * 0.3
        )

        # Clamp between 0–100
        confidence = max(0, min(100, int(confidence)))

        return JSONResponse({
            "success": True,
            "confidence": confidence,
            "emotion": analysis.get("dominant_emotion", "neutral"),
            "scores": emotions_percent,
            "method": "deepface"
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

# -------------------- Dashboard Routes -------------------- #
@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)

    name = request.session.get("user_name")
    email = request.session.get("email")
    avatar = name[0] if name else "?"

    user_interviews = list(
        interviews_collection.find(
            {"user_id": user_id},
            {
                "_id": 1,
                "timestamp": 1,
                "overallScore": 1,
                "duration": 1,
                "totalQuestions": 1,
                "accuracy": 1,
                "confidence": 1,
                "correctness": 1,
                "domain": 1
            }
        ).sort("timestamp", 1)
    )

    for i in user_interviews:
        i["_id"] = str(i["_id"])

    total_completed_interviews = len(user_interviews)
    scores = [i.get("overallScore", 0) for i in user_interviews]

    total_minutes = 0
    for i in user_interviews:
        duration = i.get("duration", "0")
        if isinstance(duration, str) and ":" in duration:
            parts = duration.split(":")
            try:
                minutes = int(parts[0])
                seconds = int(parts[1]) if len(parts) > 1 else 0
                total_minutes += minutes + seconds / 60
            except:
                pass
        else:
            try:
                total_minutes += float(duration)
            except:
                pass
    practice_hours = round(total_minutes / 60, 2)

    correctness_list = [i.get("correctness", 0) for i in user_interviews]
    total_questions_list = [i.get("totalQuestions", 0) for i in user_interviews]
    total_correct = sum(correctness_list)
    total_questions = sum(total_questions_list)
    success_rate = round((total_correct / total_questions) * 100, 2) if total_questions > 0 else 0

    overallscore = round(sum(scores) / len(scores), 2) if scores else 0

    chart_labels = [i.get("domain", f"Interview {idx+1}") for idx, i in enumerate(user_interviews)]
    chart_data = scores

    chart_labels = chart_labels or []
    chart_data = chart_data or []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "name": name,
            "email": email,
            "avatar": avatar,
            "total_completed_interviews": total_completed_interviews,
            "overallscore": overallscore,
            "practice_hours": practice_hours,
            "success_rate": success_rate,
            "chart_labels": chart_labels,
            "chart_data": chart_data,
        }
    )

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    global face_verified
    face_verified = False
    return RedirectResponse(url="/", status_code=303)

@app.get("/interview", response_class=HTMLResponse)
async def interview_page(request: Request, domain: str = Query(default="General")):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    
    return templates.TemplateResponse("interview.html", {"request": request, "domain": domain})

# -------------------- Text-to-Speech -------------------- #
@app.post("/generate_audio")
async def generate_audio(request: Request, text: str = Form(...)):
    def speak_in_background(msg):
        global engine_instance
        try:
            stop_speaking()
            engine = pyttsx3.init()
            with engine_lock:
                engine_instance = engine
            voices = engine.getProperty('voices')
            for voice in voices:
                if "male" in voice.name.lower() or "david" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 170)
            engine.say(msg)
            engine.runAndWait()
        except Exception as e:
            print("❌ Speech error:", e)
        finally:
            with engine_lock:
                engine_instance = None

    threading.Thread(target=speak_in_background, args=(text,), daemon=True).start()
    return {"success": True, "message": "Speaking in background"}

# -------------------- Browser Voice Upload Transcription -------------------- #
@app.post("/transcribe_uploaded_audio")
async def transcribe_uploaded_audio(file: UploadFile = File(...)):
    try:
        import google.ai.generativelanguage as glm

        audio_dir = os.path.join("static", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        file_name = f"recorded_{int(time.time())}.wav"
        audio_path = os.path.join(audio_dir, file_name)

        with open(audio_path, "wb") as f:
            f.write(await file.read())

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        audio_part = glm.Part(
            inline_data=glm.Blob(
                mime_type="audio/wav",
                data=audio_bytes
            )
        )

        response = model.generate_content(
            [audio_part, "Transcribe this audio into text."]
        )

        return JSONResponse({
            "success": True,
            "text": response.text.strip(),
            "file_path": audio_path
        })

    except Exception as e:
        print("❌ Transcription error:", e)
        return JSONResponse({"success": False, "error": str(e)})

# -------------------- AI Feedback -------------------- #
@app.post("/evaluate_answer")
async def evaluate_answer(question: str = Form(...), answer: str = Form(...), domain: str = Form(...)):
    try:
        prompt = f"""
        You are an expert {domain} interviewer. Evaluate this answer.
        Question: {question}
        Answer: {answer}
        Give:
        1. Correctness (Yes/No/Partial)
        2. Score (0–10)
        3. Feedback (2–3 lines)
        4. Missing key points
        """
        response = model.generate_content(prompt)
        full_feedback = response.text.strip()

        summarize_prompt = f"Summarize this interview feedback in 2 short sentences for spoken feedback:\n\n{full_feedback}"
        summary_response = model.generate_content(summarize_prompt)
        summary = summary_response.text.strip()

        def speak_feedback():
            global engine_instance
            try:
                stop_speaking()
                engine = pyttsx3.init()
                with engine_lock:
                    engine_instance = engine
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "male" in voice.name.lower() or "david" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 170)
                engine.say(summary)
                engine.runAndWait()
            except Exception as e:
                print("Speech Error:", e)
            finally:
                with engine_lock:
                    engine_instance = None

        threading.Thread(target=speak_feedback, daemon=True).start()

        return {
            "success": True,
            "feedback": full_feedback,
            "summary": summary
        }

    except Exception as e:
        print("❌ Evaluation error:", e)
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.post("/summarize_feedback")
async def summarize_feedback(feedback: str = Form(...)):
    try:
        prompt = f"Summarize this interview feedback in 2–3 short spoken sentences:\n\n{feedback}"
        response = model.generate_content(prompt)
        summary = response.text.strip()
        return {"success": True, "summary": summary}
    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------- Video Feed -------------------- #
from threading import Lock
camera_lock = Lock()

def gen_frames():
    while True:
        with camera_lock:
            if not camera.isOpened():
                camera.open(0)
            ret, frame = camera.read()

        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    if camera.isOpened():
        camera.release()
        print("🛑 Camera released on shutdown.")

# -------------------- Gemini Question Generation -------------------- #
@app.get("/get_questions")
def get_questions(domain: str = Query(...)):
    try:
        prompt = f"Generate 3 interview questions for a candidate applying for a {domain} developer role. Only return questions, no answers or explanation and please return easy questions not too difficult."
        result = model.generate_content(prompt)
        questions = [q.strip() for q in result.text.split("\n") if q.strip()]
        collection.insert_many([{"domain": domain, "text": q} for q in questions])
        return {"success": True, "questions": questions}
    except Exception as e:
        print("❌ Question generation error:", e)
        return {"success": False, "error": str(e)}

# -------------------- Interview HTML -------------------- #
@app.get("/start_interview", response_class=HTMLResponse)
async def start_interview(request: Request, domain: str = Query(default=None)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("start_interview.html", {"request": request, "domain": domain})

# -------------------- Report Routes -------------------- #
@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request, id: str = Query(default=None)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    if not id:
        return RedirectResponse(url="/index", status_code=303)
    return templates.TemplateResponse("report.html", {"request": request, "id": id})

@app.get("/get_interview_report")
async def get_interview_report(request: Request, id: str = Query(...)):
    try:
        user_id = request.session.get("user_id")
        interview = interviews_collection.find_one({'_id': ObjectId(id), 'user_id': user_id})
        if interview:
            report = {
                'date': interview.get('date'),
                'duration': interview.get('duration'),
                'totalQuestions': interview.get('totalQuestions'),
                'confidence': interview.get('confidence'),
                'accuracy': interview.get('accuracy'),
                'correctness': interview.get('correctness'),
                'overallScore': interview.get('overallScore'),
                'bestPoints': interview.get('bestPoints', []),
                'laggingPoints': interview.get('laggingPoints', []),
                'recommendations': interview.get('recommendations', []),
                'overallSummary': interview.get('overallSummary', '')
            }
            return JSONResponse({'success': True, 'report': report})
        else:
            return JSONResponse({'success': False, 'error': 'Interview not found'}, status_code=404)
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@app.get("/get_completed_interviews")
async def get_completed_interviews(request: Request):
    try:
        user_id = request.session.get("user_id")
        
        if not user_id:
            print("❌ No user_id in session")
            return JSONResponse({'success': False, 'interviews': []})
        
        print(f"✅ Fetching interviews for user: {user_id}")
        
        interviews = list(interviews_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1))
        
        print(f"✅ Found {len(interviews)} interviews")
        
        interviews_list = []
        for interview in interviews:
            interview_dict = {
                'id': str(interview['_id']),
                'date': interview.get('date', 'N/A'),
                'duration': interview.get('duration', '0:00'),
                'totalQuestions': interview.get('totalQuestions', 0),
                'overallScore': interview.get('overallScore', 0),
                'confidence': interview.get('confidence', 0),
                'accuracy': interview.get('accuracy', 0),
                'correctness': interview.get('correctness', 0),
                'domain': interview.get('domain', 'General')
            }
            interviews_list.append(interview_dict)
            print(f"  📄 Interview {interview_dict['id']}: Score {interview_dict['overallScore']}%")
        
        return JSONResponse({
            'success': True,
            'interviews': interviews_list
        })
        
    except Exception as e:
        print(f"❌ Error getting interviews: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e), 'interviews': []}, status_code=500)

@app.get("/get_interviews")
async def get_interviews(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return JSONResponse([])

    interviews = list(interviews_collection.find(
        {"user_id": user_id},
        {"_id": 1, "date": 1, "domain": 1, "accuracy": 1, "confidence": 1, "overallScore": 1}
    ))
    for i in interviews:
        i["_id"] = str(i["_id"])
    return JSONResponse(interviews)

@app.post("/save_interview")
async def save_interview(request: Request):
    try:
        data = await request.json()
        user_id = request.session.get("user_id")
        if not user_id:
            return JSONResponse({"success": False, "error": "User not logged in"}, status_code=401)

        interview_doc = {
            "user_id": user_id,
            "domain": data.get("domain", "General"),
            "questions": data.get("questions", []),
            "answers": data.get("answers", []),
            "feedback": data.get("feedback", []),
            "emotions": data.get("emotions", []),
            "confidence": data.get("confidence", 0),
            "accuracy": data.get("accuracy", 0),
            "correctness": data.get("correctness", []),
            "overallScore": data.get("overallScore", 0),
            "bestPoints": data.get("bestPoints", []),
            "laggingPoints": data.get("laggingPoints", []),
            "recommendations": data.get("recommendations", []),
            "overallSummary": data.get("overallSummary", ""),
            "totalQuestions": len(data.get("questions", [])),
            "duration": data.get("duration", "0:00"),
            'timestamp': datetime.now(),
            'date': datetime.now().strftime('%B %d, %Y'),
        }

        result = interviews_collection.insert_one(interview_doc)
        return JSONResponse({"success": True, "interview_id": str(result.inserted_id)})

    except Exception as e:
        print("❌ Error saving interview:", e)
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# -------------------- UPDATED SAVE INTERVIEW WITH GEMINI ANALYSIS -------------------- #
@app.post("/save_interview_result")
async def save_interview_result(request: Request):
    try:
        data = await request.json()
        user_id = request.session.get("user_id")

        if not user_id:
            return JSONResponse({'success': False, 'error': 'User not logged in'}, status_code=401)

        from datetime import datetime
        current_time = datetime.now()
        
        domain = data.get('domain', 'General')
        questions = data.get('questions', [])
        answers = data.get('answers', [])
        confidence = data.get('confidence', 0)
        accuracy = data.get('accuracy', 0)
        correctness = data.get('correctness', 0)
        overallScore = data.get('overallScore', 0)
        
        # Create detailed Q&A transcript for Gemini
        qa_transcript = ""
        for i, (q, a) in enumerate(zip(questions, answers)):
            qa_transcript += f"\n{'='*60}\n"
            qa_transcript += f"Question {i+1}: {q}\n"
            qa_transcript += f"Student's Answer: {a}\n"
        
        # CRITICAL: Only generate AI analysis if answers were provided
        if len(answers) > 0 and any(a.strip() for a in answers):
            print(f"🤖 Generating Gemini analysis for {len(answers)} answers...")
            
            gemini_prompt = f"""You are an expert {domain} interviewer analyzing a completed interview.

INTERVIEW DETAILS:
- Domain: {domain}
- Total Questions: {len(questions)}
- Confidence Level: {confidence}%
- Technical Accuracy: {accuracy}%
- Overall Score: {overallScore}%

COMPLETE Q&A TRANSCRIPT:
{qa_transcript}

Based on the ACTUAL answers provided above, analyze this interview performance.

CRITICAL INSTRUCTIONS:
1. Identify SPECIFIC strengths from their actual responses
2. Point out CONCRETE weaknesses in their answers
3. Give ACTIONABLE recommendations based on what they struggled with
4. Reference ACTUAL concepts/topics they discussed
5. If answers are missing or empty, state that clearly

Return ONLY valid JSON (no markdown, no backticks, no extra text):
{{
  "bestPoints": ["specific strength 1", "specific strength 2", "specific strength 3"],
  "laggingPoints": ["specific weakness 1", "specific weakness 2", "specific weakness 3"],
  "recommendations": ["actionable step 1", "actionable step 2", "actionable step 3"],
  "overallSummary": "2-3 sentences summarizing their SPECIFIC performance with concrete examples from their answers"
}}
"""

            try:
                gemini_response = model.generate_content(gemini_prompt)
                response_text = gemini_response.text.strip()
                
                print(f"📝 Gemini raw response: {response_text[:200]}...")
                
                # Clean and parse JSON
                import json
                clean_text = response_text.replace('```json', '').replace('```', '').strip()
                
                # Extract JSON if embedded in text
                json_match = clean_text
                if '{' in clean_text and '}' in clean_text:
                    start = clean_text.find('{')
                    end = clean_text.rfind('}') + 1
                    json_match = clean_text[start:end]
                
                ai_analysis = json.loads(json_match)
                
                print("✅ Gemini analysis parsed successfully")
                
                bestPoints = ai_analysis.get('bestPoints', [])
                laggingPoints = ai_analysis.get('laggingPoints', [])
                recommendations = ai_analysis.get('recommendations', [])
                overallSummary = ai_analysis.get('overallSummary', '')
                
            except Exception as e:
                print(f"⚠️ Gemini analysis failed: {e}")
                # Fallback based on scores
                if overallScore < 40:
                    bestPoints = [f"Attempted {domain} interview questions"]
                    laggingPoints = [
                        f"Needs stronger foundation in {domain} concepts",
                        "Should provide more detailed technical explanations",
                        "Requires more practice with interview questions"
                    ]
                    recommendations = [
                        f"Study core {domain} fundamentals thoroughly",
                        "Practice explaining concepts clearly",
                        "Work through coding problems daily"
                    ]
                    overallSummary = f"Completed {domain} interview with {overallScore}% score. Significant improvement needed in technical depth and explanation clarity."
                else:
                    bestPoints = [
                        f"Demonstrated {domain} knowledge",
                        "Attempted all interview questions",
                        "Showed problem-solving approach"
                    ]
                    laggingPoints = [
                        "Could provide more technical depth",
                        f"Should expand {domain} concept understanding",
                        "Communication could be more structured"
                    ]
                    recommendations = [
                        f"Practice advanced {domain} topics",
                        "Work on detailed technical explanations",
                        "Review interview best practices"
                    ]
                    overallSummary = f"Completed {domain} interview with {overallScore}% score. Shows foundational understanding but needs more depth and practice."
        
        else:
            # No answers provided - return minimal/empty analysis
            print("⚠️ No answers provided, using minimal analysis")
            bestPoints = []
            laggingPoints = [
                "No answers were submitted during the interview",
                f"Need to attempt {domain} interview questions",
                "Practice is essential for improvement"
            ]
            recommendations = [
                f"Start learning {domain} fundamentals",
                "Prepare answers for common interview questions",
                "Practice speaking and explaining concepts"
            ]
            overallSummary = f"Interview was not completed - no answers were provided. Please attempt the interview questions and provide detailed responses."
        
        # Save to database
        interview_result = {
            'user_id': user_id,
            'timestamp': current_time,
            'date': current_time.strftime('%B %d, %Y'),
            'duration': data.get('duration', '0:00'),
            'totalQuestions': len(questions),
            'confidence': confidence,
            'accuracy': accuracy,
            'correctness': correctness,
            'overallScore': overallScore,
            'bestPoints': bestPoints,
            'laggingPoints': laggingPoints,
            'recommendations': recommendations,
            'overallSummary': overallSummary,
            'questions': questions,
            'answers': answers,
            'domain': domain
        }

        result = interviews_collection.insert_one(interview_result)
        interview_id = str(result.inserted_id)
        
        print(f"✅ Interview saved: {interview_id} with {'Gemini' if answers else 'minimal'} analysis")

        return JSONResponse({
            'success': True,
            'interviewId': interview_id,
            'redirectUrl': f"/report?id={interview_id}"
        })

    except Exception as e:
        print(f"❌ Error saving interview: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

# -------------------- PDF Download -------------------- #
@app.get("/download_report_pdf")
async def download_report_pdf(request: Request, id: str = Query(...)):
    try:
        user_id = request.session.get("user_id")
        interview = interviews_collection.find_one({
            '_id': ObjectId(id),
            'user_id': user_id
        })

        if not interview:
            return JSONResponse({'error': 'Interview not found'}, status_code=404)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )

        story.append(Paragraph("Interview Performance Report", title_style))
        story.append(Spacer(1, 0.2*inch))

        details_data = [
            ['Interview Date:', interview.get('date', 'N/A')],
            ['Duration:', interview.get('duration', 'N/A')],
            ['Questions:', str(interview.get('totalQuestions', 0))],
        ]

        details_table = Table(details_data, colWidths=[2*inch, 4*inch])
        details_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f7fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(details_table)
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Performance Scores", heading_style))
        scores_data = [
            ['Metric', 'Score'],
            ['Confidence', f"{interview.get('confidence', 0)}%"],
            ['Accuracy', f"{interview.get('accuracy', 0)}%"],
            ['Correctness', f"{interview.get('correctness', 0)}%"],
            ['Overall', f"{interview.get('overallScore', 0)}%"],
        ]

        scores_table = Table(scores_data, colWidths=[3*inch, 3*inch])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Overall Summary", heading_style))
        story.append(Paragraph(interview.get('overallSummary', 'N/A'), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Only show best points if they exist
        if interview.get('bestPoints'):
            story.append(Paragraph("Strengths", heading_style))
            for point in interview.get('bestPoints', []):
                story.append(Paragraph(f"• {point}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Areas for Improvement", heading_style))
        for point in interview.get('laggingPoints', []):
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("Recommendations", heading_style))
        for rec in interview.get('recommendations', []):
            story.append(Paragraph(f"• {rec}", styles['Normal']))

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type='application/pdf',
            headers={'Content-Disposition': f'attachment; filename=interview_report_{id}.pdf'}
        )

    except Exception as e:
        print(f"❌ PDF error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=500)