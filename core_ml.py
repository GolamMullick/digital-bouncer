import cv2
import numpy as np
import tensorflow as tf
import os
import boto3

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "digital-bouncer-models")
S3_MODEL_KEY = "liveness_cnn.keras"
LOCAL_MODEL_PATH = "model_registry/liveness_cnn.keras"


def download_model_from_s3():
    """Downloads model weights from AWS S3 if not available locally."""
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("model_registry", exist_ok=True)
        print(f"⬇️ Downloading {S3_MODEL_KEY} from S3 bucket '{S3_BUCKET_NAME}'...")
        
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        print("✅ Model weights downloaded successfully from S3!")


def check_liveness(image_bytes: bytes) -> float:
    """Evaluates input selfie bytes against the Keras model stored in S3/local."""
    
    # Ensure model is fetched from S3 if missing
    download_model_from_s3()

    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
        
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image bytes.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    return float(prediction[0][0])