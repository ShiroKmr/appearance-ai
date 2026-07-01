import cv2
import mediapipe as mp
from face_validation import validateFace
from face_segmentation import faceSeg

mp_drawing = mp.solutions.drawing_utils
mp_DrawingStyles = mp.solutions.drawing_styles

def run_camera():
    mp_FaceMesh = mp.solutions.face_mesh

    with mp_FaceMesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as faceMesh:
        # Access the main camera: 0
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            # If the image capture isn't successfull, then break
            success, image = cap.read()

            if not success:
                break
            
            image = cv2.flip(image, 1)
            imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(imageRgb)

            if results.multi_face_landmarks:
                for faceLandmarks in results.multi_face_landmarks:
                    validationErrors = validateFace(image, faceLandmarks)

                    if validationErrors:
                        for index, error in enumerate(validationErrors):
                            cv2.putText(
                                image,
                                error,
                                (30, 40 + index * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                    else:
                        faceSeg(image, faceLandmarks)

            if not results.multi_face_landmarks:
                cv2.putText(
                    image,
                    "No face detected.",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Show the captured image and exit if 'q' is pressed
            cv2.imshow("My face", image)

            if cv2.waitKey(100) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()