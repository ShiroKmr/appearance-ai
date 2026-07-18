## Project description
Real-time facial analysis with color season classification and virtual makeup try-on.

## Milestones
- [x] Set a camera capture
- [x] Create a test with numpy:
    - [x] Features analysis
    - [ ] Hair color analysis
    - [x] Color season
- [ ] Train a model on the dataset
- [ ] Expand to capture the faceMesh and give out an analysis real-time
- [ ] Male-up overlay

## Tech stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Pathlib
- Torch/Torchvision
- Scikit-learn

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
Train - check the dataset, load and train EfficientNet model on it.
Model - load the best model from the train to later connect it to the app.

## To-do/improvements:
- Check for glasses
- Check vertical head position
- Check hair color
- Make eye color analysis more precise

## Dataset used for color analysis
Lorenzo Stacchio and Marina Paolanti and Francesca Spigarelli and Emanuele Frontoni,
"Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification".
