# face_database.py
import os
import face_recognition
import pickle
from pathlib import Path
import cv2
from PIL import Image
import numpy as np  # 忘れずに！

def detect_face_with_landmarks(image):
    face_locations = face_recognition.face_locations(np.array(image))
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))

    if face_locations:
        top, right, bottom, left = face_locations[0]
        face_box = (left, top, right, bottom)

        landmark_points = []
        if face_landmarks_list:
            for region in face_landmarks_list[0].values():
                landmark_points.extend(region)

        return face_box, landmark_points

    return None, None

ENCODING_FILE = "encodings.pkl"
FACE_DIR = "known_faces"

class FaceDatabase:
    def __init__(self):
        self.encodings = []
        self.labels = []
        self.load()

    def load(self):
        if os.path.exists(ENCODING_FILE):
            with open(ENCODING_FILE, "rb") as f:
                data = pickle.load(f)
                self.encodings = data["encodings"]
                self.labels = data["labels"]
        else:
            self.encodings = []
            self.labels = []

    def save(self):
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump({
                "encodings": self.encodings,
                "labels": self.labels
            }, f)

    def add_face(self, image_path, label):
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)
        if not encodings:
            raise ValueError("顔が検出されませんでした")
        encoding = encodings[0]

        # 保存
        self.encodings.append(encoding)
        self.labels.append(label)
        label_dir = os.path.join(FACE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(label_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        self.save()

    def predict(self, image_path):
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)
        if not encodings:
            return None, 0.0, image

        encoding = encodings[0]
        distances = face_recognition.face_distance(self.encodings, encoding)
        if not distances.size:
            return None, 0.0, image
        min_idx = distances.argmin()
        min_dist = distances[min_idx]
        similarity = 1.0 - min_dist  # 類似度（擬似的）
        label = self.labels[min_idx]
        return label, similarity, image
