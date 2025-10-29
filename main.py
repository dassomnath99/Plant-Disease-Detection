import requests

# Predict disease
url = 'http://127.0.0.1:8000/api/predict/'
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())