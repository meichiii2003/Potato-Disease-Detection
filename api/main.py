from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes"

MODEL = tf.keras.models.load_model("C:/Users/meich/OneDrive - Asia Pacific University/Machine Learning and Data Science Project/Potato Disease Classification/saved_models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])

    # Safely return the class name and confidence
    return {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    # Read the image
    #asyn and await are used to read the file asynchronously

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000) 