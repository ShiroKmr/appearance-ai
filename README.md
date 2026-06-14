## Project description
Real-time facial analysis with color season classification and virtual makeup try-on.

## Milestones
- [x] Set a camera capture
- [ ] Train a model to give out AT LEAST: face undertone, face contrast, eye color.
- [ ] Then expand to analyse and give out a color analysis
- [ ] Train a model to find the passing makeup

## Tech stack

- Python
- OpenCV
- MediaPipe
- NumPy

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```
## Privacy
This application runs locally. No images are uploaded or stored by default.

## Files usage
App: 
- face_landmarks.py: Opens the camera, finds landmarks
- face_validation.py: Checks the camera for clear picture to improve future model's predictions
- face_segmentation.py: Makes sure that the face mask is correct

