## Project description
Real-time facial analysis with color season classification and virtual makeup try-on.

## Milestones
- [x] Set a camera capture
- [x] Create a test with numpy:
    - [x] Features analysis
    - [ ] Hair color analysis
    - [x] Color season
- [x] Train a model on the dataset
- [x] Expand to capture the faceMesh and give out an analysis real-time
- [ ] Make-up overlay

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
Please run the application from the root directory.

## Privacy
This application runs locally. No images are uploaded or stored by default.

## To-do/improvements:
- Re-write neural season classifier
- Change the data folder
- Document the pre-processing of frames, data folder

## Dataset used for color analysis
Lorenzo Stacchio and Marina Paolanti and Francesca Spigarelli and Emanuele Frontoni,
"Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification".
