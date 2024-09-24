import os

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

# The URL of the server
url = "http://127.0.0.1:5000/syndromedetection"

# Create an instance of the MLClient object
client = MLClient(url)

# Directory containing the test images
image_dir = "/Users/mitalijuvekar/Documents/SEM3/596E/Project1/data/dataset/test"

# Get all image files from the directory
image_files = [
    f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Display available images to the user
print("Available images:")
for index, image_file in enumerate(image_files):
    print(f"{index + 1}: {image_file}")

# Ask the user to select an image by number
try:
    choice = int(input("Enter the number of the image you want to process: ")) - 1
    if choice < 0 or choice >= len(image_files):
        raise ValueError("Invalid choice.")

    selected_image = image_files[choice]
    print(f"You selected: {selected_image}")
except ValueError as e:
    print(f"Error: {e}. Please enter a valid number.")
    exit(1)

# # Create inputs list with full file paths
# inputs = [
#     {"file_path": os.path.join(image_dir, image_file)}
#     for image_file in image_files[:5]  # Limit to 5 images for this example
# ]

# Create inputs list with full file path for the selected image
inputs = [{"file_path": os.path.join(image_dir, selected_image)}]

# Specify the data type
data_type = DataTypes.IMAGE

# Send a request to the server
response = client.request(inputs, data_type)

# Print the entire response
print("Full Response:")
print(response)

# Process and print individual results
print("\nIndividual Results:")
for result in response:  # Change this line to iterate directly over response
    image_name = result["file_path"]  # Access file_path directly from result
    prediction = result["result"]["class"]
    probability = result["result"]["probability"]

    print(f"Image: {image_name}")
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")
    print("---------------------------")

# from flask_ml.flask_ml_client import MLClient
# from flask_ml.flask_ml_server.constants import DataTypes
# import os

# # The URL of the server
# url = "http://127.0.0.1:5000/syndromedetection"

# # Create an instance of the MLClient object
# client = MLClient(url)

# # Directory containing the test images
# image_dir = "/Users/mitalijuvekar/Documents/SEM3/596E/Project1/data/dataset/test"

# # Get all image files from the directory
# image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # Create inputs list with full file paths
# inputs = [
#     {"file_path": os.path.join(image_dir, image_file)}
#     for image_file in image_files[:5]  # Limit to 5 images for this example
# ]

# # Specify the data type
# data_type = DataTypes.IMAGE

# # Send a request to the server
# response = client.request(inputs, data_type)

# # Print the entire response
# print("Full Response:")
# print(response)

# # Process and print individual results
# print("\nIndividual Results:")
# for result in response['results']:
#     image_name = os.path.basename(result['file_path'])
#     prediction = result['result']['class']
#     probability = result['result']['probability']

#     print(f"Image: {image_name}")
#     print(f"Prediction: {prediction}")
#     print(f"Probability: {probability:.4f}")
#     print("---------------------------")
