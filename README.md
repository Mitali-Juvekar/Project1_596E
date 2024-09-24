This project implements a machine learning service for detecting Down syndrome from facial images using a Flask-ML server and client.

## Setup Instructions ##

1. Clone the repository:
```bash
git clone https://github.com/your-username/down-syndrome-detection.git
cd down-syndrome-detection
```

2. Create and activate a virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Important: Adjust file paths
* In src/app.py, update the path to your model file:
```bash
self.model = tf.keras.models.load_model('path/to/your/syndrome_detection_model.keras')
```
* In src/client.py, update the path to your test images:
```bash
image_dir = "/path/to/your/data/dataset/test"
```

* In src/train.py, update the path to your train dataset:
```bash
data_healthy = create_images_list('path/to/your/data/dataset/healthy/healthy')
data_down = create_images_list('path/to/your/data/dataset/downSyndrome/downSyndrome')
```

Replace these paths with the actual locations on your system where you've stored the model and test images.

## Project Structure ##

* src/app.py: Contains the server implementation using Flask-ML.
* src/client.py: Implements the client-side CLI for interacting with the server.
* src/train.py: Implements the CNN model and its training parameters.
* syndrome_detection_model.keras: The trained machine learning model (ensure this file is present in the project root or update its path in app.py).
* data/dataset/test: Directory containing test images (update this path in client.py if your images are stored elsewhere).

## Using the CLI ##

1. When you run client.py, it will display a list of available images in the test images directory you specified.

2. Enter the number corresponding to the image you want to process.

3. The client will send the selected image to the server and display the prediction results.

## Expected Output ##

The client will display the following information for the processed image:

* Image filename
* Prediction (healthy or down syndrome)
* Probability of the prediction
