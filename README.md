# FitVision: Computer Vision for Exercise Performance Assessment and Injury Prevention

Poor form and lack of coaching are leading causes of injury among beginner gym-goers, and most people can't afford a personal coach. FitVision lets you upload a side-view video of a set and get feedback on every repetition, so you can spot and fix problems before they cause injury.

## How it works

- Pose estimation with Google's MediaPipe BlazePose (33 body landmarks), plus SpinePose for detailed spine analysis in deadlifts
- Exercise-specific algorithms measure joint angles, depth ratios and body alignment, and a state machine detects and counts each rep
- Each rep gets a score and a per-metric breakdown. An LLM (LLaMA 3.1 via Groq) then writes a coaching summary for the whole set
- Faces are blurred and raw uploads are deleted after processing. The output is an annotated video

## Supported exercises

- **Squat:** depth, thigh angle, knee range of motion, forward lean, heel lift
- **Push-up:** elbow depth, hip pike/sag, shoulder-over-wrist alignment, neck alignment, tempo
- **Deadlift:** spine curvature, hip hinge depth, bar path, knee lockout, tempo

**Stack:** Python (OpenCV, MediaPipe, SpinePose), Django REST Framework with JWT auth, React Native (Expo)

---

## Getting started

The project consists of three parts: the analysis model (Python), the backend (Django), and the mobile frontend (React Native / Expo).

### Setup

Python version: 3.12

Install dependencies:

```bash
pip install -r requirements.txt
```

You also need ffmpeg installed on your system (`brew install ffmpeg` on Mac).

SpinePose is required for deadlift spine curvature analysis:

```bash
pip install spinepose
```

Create a `.env` file in the root directory with:

```
GROQ_API_KEY=your_key_here
```

### Running the model standalone

```bash
cd model
python exercise_video_analysis.py path/to/video.mov squat
```

Supported exercise types: `squat`, `pushup`, `deadlift`

### Running the backend

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

### Running the frontend

```bash
cd frontend
npm install
npx expo start
```

Scan the QR code with Expo Go on your phone.
The frontend expects the backend running at the URL defined in `src/api/client.js`.

### Notes

- The backend `BASE_URL` in `frontend/src/api/client.js` needs to point to the machine's local IP if testing on a real device.
- Raw input videos are deleted after anonymisation. Only anonymised versions are stored.
