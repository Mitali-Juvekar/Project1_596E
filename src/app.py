# # src/app.py

import tensorflow as tf
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ImageResult, ResponseModel


# Load your trained model
class SyndromeDetectionModel:
    def __init__(self):
        self.model = tf.keras.models.load_model("syndrome_detection_model.keras")
        self.disease_classes = {0: "healthy", 1: "down"}

    def load_and_preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)  # Adjust if using other formats
        img = tf.image.resize(img, size=(250, 250))  # Resize to match model input shape
        img = img / 255.0  # Normalize to [0, 1]
        return img

    def predict(self, image_paths: list) -> list:
        # Load and preprocess all images
        images = [self.load_and_preprocess_image(path) for path in image_paths]

        # Convert list of tensors to a single tensor
        images_tensor = tf.stack(images)

        # Make predictions
        predictions = self.model.predict(images_tensor)
        results = []
        for pred in predictions:
            probability = pred[0]
            class_prediction = self.disease_classes[round(probability)]
            results.append((probability, class_prediction))
        return results


# Create an instance of the model
model = SyndromeDetectionModel()

# Create a server
server = MLServer(__name__)


# Create an endpoint
@server.route("/syndromedetection", DataTypes.IMAGE)
def process_image(inputs: list, parameters: dict) -> dict:
    print("Received inputs:", inputs)

    if isinstance(inputs, list):
        image_paths = [
            input_item.file_path for input_item in inputs
        ]  # Use dot notation
    else:
        return {
            "error": "Invalid input format"
        }, 400  # Return an error response if format is incorrect

    print("Image paths for prediction:", image_paths)

    results = model.predict(image_paths)

    # Create ImageResult objects using 'file_path' instead of 'image'
    results_formatted = [
        ImageResult(
            file_path=path, result={"probability": float(prob), "class": class_pred}
        )
        for path, (prob, class_pred) in zip(image_paths, results)
    ]

    response = ResponseModel(results=results_formatted)
    return response.get_response()


# Run the server
server.run()

# from flask_ml.flask_ml_server import MLServer
# from flask_ml.flask_ml_server.constants import DataTypes
# from flask_ml.flask_ml_server.models import ResponseModel, ImageResult
# import tensorflow as tf

# # Load your trained model
# class SyndromeDetectionModel:
#     def __init__(self):
#         self.model = tf.keras.models.load_model('syndrome_detection_model.keras')
#         self.disease_classes = {0: 'healthy', 1: 'down'}

#     def predict(self, image_paths: list) -> list:
#         # The model should be set up to accept file paths directly
#         predictions = self.model.predict(image_paths)
#         results = []
#         for pred in predictions:
#             probability = pred[0]
#             class_prediction = self.disease_classes[round(probability)]
#             results.append((probability, class_prediction))
#         return results

# # Create an instance of the model
# model = SyndromeDetectionModel()

# # Create a server
# server = MLServer(__name__)

# Create an endpoint
# @server.route("/syndromedetection", DataTypes.IMAGE)
# def process_image(inputs: list, parameters: dict) -> dict:
#     image_paths = [input_item['file_path'] for input_item in inputs]
#     results = model.predict(image_paths)

#     # Create ImageResult objects
#     results_formatted = [ImageResult(image=path, result={'probability': float(prob), 'class': class_pred})
#                for path, (prob, class_pred) in zip(image_paths, results)]


#     response = ResponseModel(results=results_formatted)
#     return response.get_response()
@server.route("/syndromedetection", DataTypes.IMAGE)
def process_image(inputs: list, parameters: dict) -> dict:
    print("Received inputs:", inputs)

    # Accessing the file_path attribute of FileInput objects
    if isinstance(inputs, list):
        image_paths = [
            input_item.file_path for input_item in inputs
        ]  # Use dot notation
    else:
        return {
            "error": "Invalid input format"
        }, 400  # Return an error response if format is incorrect

    # Check what image_paths looks like
    print("Image paths for prediction:", image_paths)

    # Ensure that the model's predict method is called correctly
    results = model.predict(image_paths)

    # Create ImageResult objects using 'file_path' instead of 'image'
    results_formatted = [
        ImageResult(
            file_path=path, result={"probability": float(prob), "class": class_pred}
        )
        for path, (prob, class_pred) in zip(image_paths, results)
    ]

    response = ResponseModel(results=results_formatted)
    return response.get_response()


# Run the server
server.run()


# from flask_ml.flask_ml_server import MLServer
# from flask_ml.flask_ml_server.constants import DataTypes
# from flask_ml.flask_ml_server.models import ResponseModel, ImageResult
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io

# # Load the model
# model = tf.keras.models.load_model('syndrome_detection_model.keras')

# # Create a server
# server = MLServer(__name__)

# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes))
#     img = img.resize((250, 250))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# @server.route("/predict", DataTypes.IMAGE)
# def process_image(inputs: list, parameters: dict) -> dict:
#     results = []
#     for input_data in inputs:
#         image_bytes = input_data['image']
#         processed_image = preprocess_image(image_bytes)
#         prediction = model.predict(processed_image)

#         result = "Down syndrome" if prediction[0][0] > 0.5 else "No Down syndrome"
#         confidence = float(prediction[0][0]) if result == "Down syndrome" else 1 - float(prediction[0][0])

#         results.append(ImageResult(image=image_bytes, result=result, confidence=confidence))

#     response = ResponseModel(results=results)
#     return response.get_response()

# if __name__ == '__main__':
#     server.run(debug=True)
