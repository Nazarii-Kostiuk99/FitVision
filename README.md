FitVision - COMPUTER VISION FOR EXERCISE PERFORMANCE ASSESSMENT AND INJURY PREVENTION
BSc Computer Science Final Year Project, University of Surrey
Nazarii Kostiuk

The project consists of three parts: the analysis model (Python), the backend (Django), and the mobile frontend (React Native / Expo).

---

Setup

Python version: 3.12

Install dependencies:
pip install -r requirements.txt

You also need ffmpeg installed on your system (brew install ffmpeg on Mac).

SpinePose is required for deadlift spine curvature analysis:
pip install spinepose

Create a .env file in the root directory with:
GROQ_API_KEY=your_key_here

---

Running the model standalone

cd model
python exercise_video_analysis.py path/to/video.mov squat

Supported exercise types: squat, pushup, deadlift

---

Running the backend

cd backend
python manage.py migrate
python manage.py runserver

---

Running the frontend

cd frontend
npm install
npx expo start

Scan the QR code with Expo Go on your phone.
The frontend expects the backend running at the URL defined in src/api/client.js.

---

Notes

- The backend BASE_URL in frontend/src/api/client.js needs to point to the machine's local IP if testing on a real device.
- Raw input videos are deleted after anonymisation. Only anonymised versions are stored.
