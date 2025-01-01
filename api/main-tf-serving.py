from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    " http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint="http://localhost:8501/v1/models/potatoes_model:predict"

MODEL_PATH = "../saved_models/1"
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am alive"

def read_file_as_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Use the tf.function for predictions
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 256, 256, 3], dtype=tf.float32)])
def predict_fn(inputs):
    return MODEL(inputs, training=False)

def extract_predictions(predictions):
    if isinstance(predictions, dict):
        return predictions.get('dense_1', None)
    return predictions

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data ={
        "instance": img_batch.tolist()
    }

    # Use the tf.function for predictions
    predictions = predict_fn(img_batch)
    response = requests.post(endpoint, json= json_data)

    # Print the raw predictions for debugging
    print("Raw Predictions:", predictions)

    # Extract predictions
    class_probabilities = extract_predictions(predictions)

    if class_probabilities is not None:
        predicted_class = CLASS_NAMES[np.argmax(class_probabilities)]
        confidence = float(np.max(class_probabilities))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    else:
        return {
            'error': 'Invalid format for predictions'
        }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
