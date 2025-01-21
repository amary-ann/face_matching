import cv2
import time
import mediapipe as mp
from deepface import DeepFace


class FaceRecognition:
    def __init__(self, db_path, known_distance=50.0, known_width=14.0):
        self.db_path = db_path
        self.known_distance = known_distance
        self.known_width = known_width
        self.focal_length = None
        self.last_recognition_time = 0
        self.mp_face_mesh = mp.solutions.face_mesh

    def find_focal_length(self, measured_distance, real_width, width_in_frame):
        return (width_in_frame * measured_distance) / real_width

    def calculate_distance(self, focal_length, real_width, width_in_frame):
        return (real_width * focal_length) / width_in_frame

    def recognize_face(self, frame):
        return DeepFace.find(frame, db_path=self.db_path, enforce_detection=False, model_name="Facenet")

    def process_frame(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # Perform face recognition
        start_time = time.perf_counter()
        results = self.recognize_face(frame_rgb)
        end_time = time.perf_counter()
        time_elapsed = end_time - start_time

        if results and len(results[0]['identity']) > 0:
            name = results[0]['identity'][0].split('/')[2].split('\\')[0]
            xmin = int(results[0]['source_x'][0])
            ymin = int(results[0]['source_y'][0])

            width = results[0]['source_w'][0]
            height = results[0]['source_h'][0]

            xmax = int(xmin + width)
            ymax = int(ymin + height)

            threshold = results[0]['threshold'][0]

            # Calculate focal length if not already set
            if self.focal_length is None:
                self.focal_length = self.find_focal_length(self.known_distance, self.known_width, width)

            # Calculate distance to the face
            distance = self.calculate_distance(self.focal_length, self.known_width, width)

            # Draw bounding box and annotations
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (90, 215, 200), 2)
            cv2.putText(frame, f"Name: {name}", (xmin, ymin - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time elapsed: {time_elapsed:.6f}s", (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Threshold: {threshold}", (xmin, ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display model name
        cv2.putText(frame, "Model: FACENET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 30, 15), 2)

        return frame

    def process_video(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Process the frame
            processed_frame = self.process_frame(frame)

            # Display the video feed
            cv2.imshow('Face Recognition', processed_frame)

            # Exit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    db_path = "./Database/"
    face_recognition = FaceRecognition(db_path=db_path)
    face_recognition.process_video()
