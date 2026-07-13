import cv2
import threading
import time
import speech_recognition as sr
from textblob import TextBlob
from deepface import DeepFace
from gtts import gTTS
from playsound import playsound
import os
import uuid

# --------------- GLOBALS ----------------
r = sr.Recognizer()
current_emotion = "Neutral"
interview_active = True

# --------------- SPEAK (Perfect Sync) ----------------
def speak(text):
    print("Bot:", text)   # Text instantly printed

    filename = f"voice_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    playsound(filename)  # Audio plays immediately
    os.remove(filename)

# --------------- LISTEN ----------------
def listen():
    with sr.Microphone() as source:
        speak("You can answer now.")
        print("🎤 Listening...")

        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=12)
            text = r.recognize_google(audio)
            print("You:", text)
            return text
        except:
            speak("I couldn't hear anything. Please try again.")
            return ""

# --------------- CAMERA EMOTION THREAD ----------------
def camera_thread():
    global current_emotion, interview_active

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not found. Continuing without emotion analysis.")
        return

    # ❌ Removed speak() here to prevent overlap

    while interview_active:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]["dominant_emotion"]
        except:
            pass

        cv2.putText(frame, f"Emotion: {current_emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AI Interview Coach - Press Q to stop", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            interview_active = False
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------- ANALYZE ANSWER ----------------
def analyze_answer(answer):
    blob = TextBlob(answer)
    sentiment = blob.sentiment.polarity
    words = len(answer.split())

    fb = []

    if words < 8:
        fb.append("Try giving a more detailed answer.")

    if sentiment > 0.3:
        fb.append("Your answer sounds positive, great job!")
    elif sentiment < 0:
        fb.append("Try to sound a bit more confident.")

    if current_emotion.lower() in ["sad", "angry", "fear"]:
        fb.append("Try to relax and maintain a gentle smile.")
    else:
        fb.append("Your facial expression looks confident!")

    return fb

# --------------- QUESTIONS ----------------
questions = [
    "Tell me about yourself.",
    "Why did you choose this career?",
    "What are your strengths and weaknesses?",
    "Describe a challenge you faced and how you solved it.",
    "Where do you see yourself in five years?",
    "Why should we hire you?",
    "Can you describe a time you worked in a team?",
    "Tell me about a time you demonstrated leadership.",
    "What motivates you in a professional environment?",
    "How do you handle stress or pressure?",
    "What is your biggest achievement?",
    "What skill are you currently improving?",
    "How do you deal with failure?",
    "What kind of work environment do you prefer?",
    "Do you have any questions for us?"
]

# --------------- INTERVIEW ----------------
def start_interview():
    global interview_active

    speak("Hello! Welcome to your AI Interview Coach.")
    speak("Your interview practice will begin now.")

    for q in questions:
        if not interview_active:
            break

        speak(q)
        answer = listen()

        if answer.strip() == "":
            speak("Let's continue to the next question.")
            continue

        feedback = analyze_answer(answer)

        speak("Here is your feedback:")
        for f in feedback:
            speak(f)

    speak("Your interview session is complete. Good job!")
    interview_active = False

# --------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()

    time.sleep(2)  # Wait for camera to initialize
    speak("Camera is ON. I am analysing your expressions.")

    start_interview()
    time.sleep(2)