## Project description
Real-time facial analysis with color season classification.

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
python -m app.main
```
Please run the application from the root directory.

## Files
App: Real-time face features analysis using a ResNet18 neural network
App/Baseline: Manual analysis and season prediction

ResNet and ResNet_train were split to separate the training from the model's definition. 

## Privacy
This application runs locally. No images are uploaded or stored by default.

## Dataset used for color analysis
Lorenzo Stacchio and Marina Paolanti and Francesca Spigarelli and Emanuele Frontoni,
"Deep Armocromia: A Novel Dataset for Face Seasonal Color Analysis and Classification".
