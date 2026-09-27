import cv2
import mediapipe as mp

from face_validation import validateFace
from neural_season_classifier import SeasonPredictor, drawSeasonPrediction

def runCamera():
    faceMeshModule = mp.solutions.face_mesh
    seasonPredictor = SeasonPredictor()
    cameraCapture = cv2.VideoCapture(0)
    currentPrediction = None
    missingFrameCount = 0

    if not cameraCapture.isOpened():
        seasonPredictor.close()
        raise RuntimeError("The camera could not be opened.")

    try:
        with faceMeshModule.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,min_tracking_confidence=0.5) as faceMesh:
            while cameraCapture.isOpened():
                success, image = cameraCapture.read()

                if not success:
                    break

                image = cv2.flip(image, 1)
                imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = faceMesh.process(imageRgb)

                if results.multi_face_landmarks:
                    missingFrameCount = 0
                    faceLandmarks = results.multi_face_landmarks[0]
                    validationErrors = validateFace(image, faceLandmarks)

                    if validationErrors:
                        for index, error in enumerate(validationErrors):
                            cv2.putText(
                                image,
                                error,
                                (30, 95 + index * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                    else:
                        # Valid frames are periodically classified while the latest stable result remains visible between inferences.
                        currentPrediction = seasonPredictor.processFrame(image,imageRgb,faceLandmarks)
                        drawSeasonPrediction(image, currentPrediction)
                else:
                    missingFrameCount += 1

                    if missingFrameCount >= 15:
                        seasonPredictor.reset()
                        currentPrediction = None

                    cv2.putText(
                        image,
                        "No face detected.",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Color season analysis", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cameraCapture.release()
        seasonPredictor.close()
        cv2.destroyAllWindows()
