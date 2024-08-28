# from .views import predict_view
from django.urls import path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from rest_framework.response import Response
from rest_framework.decorators import api_view
import cv2
import urllib

import io
import cloudinary.api
import cloudinary.uploader
import cloudinary


def normalized_image(img):
    mean = 159.88
    std_dev = 46.25
    normalized_image = (img - mean) / std_dev
    # normalized_image = (resized_image) / 255

    processed_image = np.expand_dims(normalized_image, axis=0)
    return processed_image


def otsu_threshold(img_gray):
    ret2, th2 = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(th2)
    return th2


def adaptive_threshold(img_gray):
    th3 = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th3


def image_gradients(img_gray):
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
    return sobelx, sobely


def canny(img_gray):
    edges = cv2.Canny(img_gray, 10, 100)
    return edges


def clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    cl1 = clahe.apply(img_gray)
    return cl1


def return_all_images(img_path):

    req = urllib.request.urlopen(img_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    otsu_img = otsu_threshold(img_gray)
    adap_img = adaptive_threshold(img_gray)
    sobel1, sobel2 = image_gradients(img_gray)
    edge_img = canny(img_gray)
    clahe_img = clahe(img_gray)

    return [otsu_img, adap_img, sobel1, sobel2, edge_img, clahe_img]


def upload_numpy_array_as_image(numpy_array):
    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(numpy_array)
    pil_image = pil_image.convert("P")

    # Convert the PIL image to bytes
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Configure Cloudinary with credentials from settings.py
    cloudinary.config(
        cloud_name="dctrj0w4q",
        api_key="767661174254334",
        api_secret="ytC7BiTjEa9jigTGb8DgOYzbDt8"
    )

    # Upload the image to Cloudinary
    response = cloudinary.uploader.upload(image_bytes, resource_type="image")

    # Return the public URL of the uploaded image
    return response["secure_url"]


def upload_all_images(img_path):
    img_list = np.array(return_all_images(img_path))
    # print(img_list.shape, img_list)
    url_list = []
    for numpy_arr in img_list:
        url_list.append(upload_numpy_array_as_image(numpy_arr))

    return url_list


@csrf_exempt
@api_view(['POST'])
def predict_view(request):
    lesion_type_dict = {
        0: 'Actinic keratoses',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions ',
        3: 'Dermatofibroma',
        4: 'Melanoma',
        5: 'Melanocytic nevi',
        6: 'Vascular lesions'
    }

    def process_image_from_url(image_url, mean, std_dev):
        # Download the image from the URL
        response = requests.get(image_url)

        if response.status_code == 200:
            # Read the image from the response content
            image_bytes = BytesIO(response.content)
            original_image = cv2.imdecode(
                np.frombuffer(image_bytes.read(), np.uint8), -1)
            original_RGB = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Resize the image to (100, 75)
            resized_image = cv2.resize(original_RGB, (120, 90))
            # Add batch dimension
            normalized_image = (resized_image - mean) / std_dev
            # normalized_image = (resized_image) / 255

            processed_image = np.expand_dims(normalized_image, axis=0)
            return processed_image
        else:
            print(f"Failed to download image from {image_url}")
            return None

    if request.method == 'POST':

        # Load the pre-trained TensorFlow model
        model = tf.keras.models.load_model(
            './btp_melanoma_app/skin_cancer_detection7_mean_std.h5')

        data = json.loads(request.body.decode('utf-8'))
        url_link = data.get('url')

        mean_value = 159.88
        std_dev_value = 46.25
        reshaped_image = process_image_from_url(
            url_link, mean_value, std_dev_value)

        # url_links = requests.get(url_link)

        # image_content = url_links.content

        # # Read the image file and convert it to a NumPy array
        # image = Image.open(BytesIO(image_content))
        # image = image.resize((75, 100, 3))
        # image_array = np.asarray(image)

        # # Normalize the image data (if required)
        # # Adjust this based on your model's requirements
        # normalized_image = image_array

        # # Reshape the image data if necessary
        # # Adjust dimensions based on your model
        # reshaped_image = np.reshape(normalized_image, (1, 75, 100, 3))

        # Make predictions using the loaded model
        predictions = model.predict(reshaped_image)
        # print('predictions: ', predictions)

        pred_classes = np.argmax(predictions, axis=1)
        # print('pred_classes: ', pred_classes)

        # Process the predictions and prepare the response
        result = lesion_type_dict[int(pred_classes)]
        print(result)

        # filter code
        filter_url_list = [url_link]
        filter_url_list.extend(upload_all_images(url_link))
        print(filter_url_list)

        # Return the result as a JSON response
        return Response({"result": result,
                         "url_list": filter_url_list})

    else:
        return Response({'error': 'Invalid request method'})


# urls.py
# urlpatterns = [
#     path('predict/', predict_view, name='predict'),
# ]
