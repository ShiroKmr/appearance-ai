## Project description
Real-time facial analysis with color season classification and virtual makeup try-on.

## Milestones
- [x] Set a camera capture
- [ ] Create a test with numpy:
    - [x] Features analysis
    - [ ] Color season
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
- camera.py: Opens the camera, finds landmarks
- face_validation.py: Checks the camera for clear picture to improve future model's predictions
- face_segmentation.py: Color analysis
- season_classifier.py: Takes face_segmentation results and outputs the color season

## To-do/improvements:
- Check for glasses
- Check vertical head position
- Check hair color
- Make eye color analysis more precise