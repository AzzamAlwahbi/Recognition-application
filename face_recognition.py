import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# ====== SETTINGS ======
DATASET_PATH = "dataset"
MODEL_PATH = "model"
# ======================

# Create model directory if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def extract_face_features(img_path):
    """Detect face and return its features"""
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
        
    # Use the first face found
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to consistent size for feature extraction
    resized_face = cv2.resize(face_roi, (100, 100))
    
    # Flatten image to create feature vector
    return resized_face.flatten()

def train_model():
    print("Training face recognizer...")
    features = []
    labels = []
    label_dict = {}
    
    # Get all person folders
    person_folders = [f for f in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, f))]
    
    # Process each person
    for person_id, person_name in enumerate(person_folders):
        label_dict[person_id] = person_name
        person_dir = os.path.join(DATASET_PATH, person_name)
        
        # Process each image in person's folder
        for img_name in os.listdir(person_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, img_name)
                face_features = extract_face_features(img_path)
                
                if face_features is not None:
                    features.append(face_features)
                    labels.append(person_id)
                    print(f"Processed {img_path}")
                else:
                    print(f"No face found in {img_path}")
    
    if not features:
        print("Error: No faces found in dataset!")
        return None, None
    
    # Create and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)
    
    # Save model and label dictionary
    joblib.dump(knn, os.path.join(MODEL_PATH, "face_model.pkl"))
    joblib.dump(label_dict, os.path.join(MODEL_PATH, "label_dict.pkl"))
    
    print(f"Training complete! Model saved to {MODEL_PATH}/")
    return knn, label_dict

def recognize_face(image_path):
    # Load model if not provided
    if not hasattr(recognize_face, 'knn') or not hasattr(recognize_face, 'label_dict'):
        try:
            recognize_face.knn = joblib.load(os.path.join(MODEL_PATH, "face_model.pkl"))
            recognize_face.label_dict = joblib.load(os.path.join(MODEL_PATH, "label_dict.pkl"))
        except:
            print("Model not found. Training first...")
            recognize_face.knn, recognize_face.label_dict = train_model()
            if recognize_face.knn is None:
                return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open image at {image_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face features
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (100, 100))
        face_features = resized_face.flatten().reshape(1, -1)
        
        # Predict using KNN
        prediction = recognize_face.knn.predict(face_features)
        confidence = recognize_face.knn.predict_proba(face_features).max()
        
        # Get person name
        person_id = prediction[0]
        name = recognize_face.label_dict.get(person_id, "Unknown")
        
        # Draw rectangle and name
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"{confidence*100:.1f}%", (x, y+h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow('Recognition Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ====== RUN THE PROGRAM ======
if __name__ == "__main__":
    # Train the model
    train_model()
    
    # Recognize a new photo (change this to your test image)
    test_photo = "test.jpg"
    recognize_face(test_photo)