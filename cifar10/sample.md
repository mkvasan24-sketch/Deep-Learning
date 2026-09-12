# Build a Production-Quality CIFAR-10 Image Classification Web App Using Flask

Act as a senior Python developer, ML engineer, Flask developer, and UI/UX designer.

I have trained a CIFAR-10 image classification model using Keras/TensorFlow. I want you to build a complete, beautiful, responsive web application around this trained model using the **Flask framework**.

The application must allow a user to:

1. Upload an image.
2. Preview the uploaded image.
3. Click a **Predict** button.
4. Preprocess the image exactly as required by the trained CIFAR-10 model.
5. Run the trained model.
6. Display the predicted CIFAR-10 class.
7. Display the prediction confidence/probability.
8. Show the top predictions if possible.
9. Clear/reset the uploaded image and prediction.
10. Upload another image and predict again.

Do not build only a prototype UI. Build a complete working Flask project with frontend, backend, model loading, preprocessing, prediction, error handling, and clean project organization.

---

# 1. EXISTING MACHINE LEARNING MODEL

The model was trained using the Keras CIFAR-10 dataset.

The training code uses:

```python
import keras
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Flatten, Dense
```

The CIFAR-10 dataset is loaded using:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

The training and testing images are normalized using:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

The trained model architecture is:

```python
model = Sequential()

model.add(Flatten(input_shape=(32,32,3)))

model.add(Dense(
    units=312,
    activation='relu'
))

model.add(Dense(
    units=256,
    activation='relu'
))

model.add(Dense(
    units=128,
    activation='relu'
))

model.add(Dense(
    units=10,
    activation='softmax'
))
```

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

The model was trained with:

```python
model.fit(
    x=x_train,
    y=y_train,
    batch_size=70,
    epochs=10,
    validation_split=0.20,
    verbose=2
)
```

The CIFAR-10 class names are:

```python
class_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
```

IMPORTANT:

The web application must use the **same preprocessing logic as training**.

For uploaded images:

1. Read the image.
2. Convert it to RGB.
3. Resize it to exactly `32 × 32` pixels.
4. Convert it to a NumPy array.
5. Convert pixel values to floating point.
6. Normalize pixel values using:

```python
image = image / 255.0
```

7. Add the batch dimension:

```python
image = np.expand_dims(image, axis=0)
```

The final input to the model must have shape:

```text
(1, 32, 32, 3)
```

Do NOT accidentally flatten the image manually before prediction because the model's `Flatten()` layer already performs this operation.

---

# 2. MODEL FILE

First inspect the existing project/workspace and determine whether the trained model is already saved.

If a trained model file exists, use it.

Preferred formats:

```text
.keras
.h5
```

For example:

```text
cifar10_model.keras
```

If the model is not currently saved, create a clear instruction/script for saving the trained model, for example:

```python
model.save("cifar10_model.keras")
```

Do not retrain the model every time the Flask application starts.

The Flask application should load the model **once when the server starts**, not once for every prediction request.

Use TensorFlow/Keras model loading appropriately.

Example concept:

```python
from tensorflow.keras.models import load_model

model = load_model("cifar10_model.keras")
```

Adapt this to the actual installed Keras/TensorFlow version if necessary.

---

# 3. APPLICATION TECHNOLOGY STACK

Use:

### Backend

* Python
* Flask
* TensorFlow/Keras
* NumPy
* Pillow/PIL

### Frontend

* HTML5
* CSS3
* JavaScript
* Modern responsive UI

You may use a lightweight CSS framework only if it improves the design, but avoid unnecessary dependencies.

The application must work on:

* Desktop
* Laptop
* Tablet
* Mobile

---

# 4. PROJECT STRUCTURE

Create a clean project structure similar to:

```text
cifar10-flask-app/
│
├── app.py
├── requirements.txt
├── README.md
├── cifar10_model.keras
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── script.js
│   │
│   └── images/
│
└── uploads/
```

If a better Flask structure is appropriate, you may improve it while keeping it beginner-friendly.

---

# 5. FLASK BACKEND

Create a Flask application.

The main route should be:

```text
GET /
```

It should render:

```text
index.html
```

Create a prediction endpoint such as:

```text
POST /predict
```

The frontend should send the uploaded image to `/predict`.

The backend should:

1. Validate that an image was uploaded.
2. Validate the file type.
3. Safely process the uploaded file.
4. Convert the image to RGB.
5. Resize to 32×32.
6. Normalize pixel values.
7. Add batch dimension.
8. Run:

```python
predictions = model.predict(image)
```

9. Determine the class with the highest probability.
10. Convert the class index into the corresponding CIFAR-10 class name.
11. Return JSON to the frontend.

Example response structure:

```json
{
    "success": true,
    "prediction": "cat",
    "confidence": 87.42,
    "top_predictions": [
        {
            "class": "cat",
            "confidence": 87.42
        },
        {
            "class": "dog",
            "confidence": 8.13
        },
        {
            "class": "frog",
            "confidence": 2.21
        }
    ]
}
```

The exact implementation can be improved as necessary.

---

# 6. IMAGE UPLOAD REQUIREMENTS

The upload interface should support:

* Click to upload
* Drag and drop
* Image preview
* Replace image
* Clear image
* Predict button

Supported formats:

```text
JPG
JPEG
PNG
WEBP
```

Set a reasonable maximum file size.

Do not allow arbitrary dangerous file uploads.

The user should receive a friendly error if:

* No image was selected.
* Unsupported file format is selected.
* File is too large.
* Image cannot be decoded.
* Prediction fails.

Never expose Python stack traces to the user.

---

# 7. MAIN USER INTERFACE

Create a visually impressive modern ML dashboard.

The design should feel like a polished AI image-classification product rather than a basic student Flask project.

Suggested application title:

# CIFAR Vision

Subtitle:

```text
AI-Powered CIFAR-10 Image Classification
```

The overall design should be:

* Modern
* Minimal
* Professional
* Elegant
* Tech/AI inspired
* Easy to understand
* Beginner-friendly
* Responsive

Use a visually attractive layout with:

* Soft gradients
* Clean cards
* Subtle shadows
* Rounded corners
* Smooth transitions
* Good typography
* Proper spacing
* Attractive buttons
* Clear visual hierarchy

Do not overcrowd the interface.

---

# 8. HERO SECTION

At the top of the page, create a hero section.

Example:

```text
CIFAR Vision
See what AI sees.

Upload an image and let your trained neural network identify
which CIFAR-10 category it belongs to.
```

Include a small badge:

```text
● AI IMAGE CLASSIFIER
```

Add a short technical indicator:

```text
CIFAR-10 • 10 Classes • Keras Model
```

---

# 9. UPLOAD CARD

Create a large central upload card.

Before uploading, show:

```text
Upload an image

Drag & drop your image here
or

Browse files

Supported formats: JPG, PNG, WEBP
Maximum size: 10 MB
```

Use an attractive upload icon.

The drag-and-drop area should respond visually when a file is dragged over it.

After selecting an image:

* Hide or reduce the empty upload state.
* Display the image preview.
* Show filename.
* Show image dimensions if available.
* Show a remove/clear button.

---

# 10. IMAGE PREVIEW

The uploaded image should appear in a beautiful preview container.

Example layout:

```text
┌─────────────────────────────┐
│                             │
│       IMAGE PREVIEW         │
│                             │
│        uploaded image       │
│                             │
└─────────────────────────────┘

filename.jpg
800 × 600
```

Even though CIFAR-10 internally uses 32×32 input, do not make the uploaded image visually tiny.

The frontend should display the original uploaded image.

The backend should resize a copy to 32×32 for inference.

---

# 11. PREDICT BUTTON

Create a prominent button:

```text
✨ Predict Image
```

The button should:

* Be disabled until an image is selected.
* Show a loading state after clicking.
* Prevent duplicate requests while prediction is running.

Loading state:

```text
Analyzing image...
```

or:

```text
AI is thinking...
```

Use a subtle animated spinner.

---

# 12. CLEAR BUTTON

Create a secondary button:

```text
Clear
```

When clicked, it must completely reset:

* Uploaded image
* Preview
* Filename
* Prediction
* Confidence
* Top predictions
* Error messages
* Loading state

The user should then be able to upload another image immediately.

---

# 13. PREDICTION RESULT CARD

After prediction, display a large result card.

Example:

```text
PREDICTION

🐱

CAT

87.42%
Confidence
```

Make the predicted class visually dominant.

Use an appropriate icon/emoji for each class if desired.

Suggested mapping:

```text
airplane   → ✈️
automobile → 🚗
bird       → 🐦
cat        → 🐱
deer       → 🦌
dog        → 🐶
frog       → 🐸
horse      → 🐴
ship       → 🚢
truck      → 🚚
```

Do not rely only on emoji; the actual predicted class name must be clearly displayed.

---

# 14. CONFIDENCE DISPLAY

Show the confidence as both:

1. Percentage
2. Visual progress bar/ring

Example:

```text
87.42%

█████████████████░░░

Model confidence
```

Animate the confidence bar when the prediction appears.

Format confidence to two decimal places.

---

# 15. TOP 3 PREDICTIONS

Display the three highest probability classes.

Example:

```text
Top Predictions

1. Cat          87.42%
   ███████████████████

2. Dog           8.13%
   ███

3. Frog          2.21%
   ██
```

Sort the predictions from highest to lowest.

Do not simply display the first three output neurons unless they are actually the top three probabilities.

Use NumPy sorting such as:

```python
np.argsort(predictions[0])[::-1]
```

Then select the top 3.

---

# 16. CIFAR-10 CLASS INFORMATION

Add a small section explaining the classes the model can recognize.

Display all 10:

```text
✈️ Airplane
🚗 Automobile
🐦 Bird
🐱 Cat
🦌 Deer
🐶 Dog
🐸 Frog
🐴 Horse
🚢 Ship
🚚 Truck
```

This helps users understand the model's limitations.

Add a note:

```text
This model is trained on CIFAR-10 and is designed to classify
images into these 10 categories.
```

---

# 17. IMPORTANT MODEL LIMITATION

Clearly communicate that the model is a CIFAR-10 classifier.

For example:

```text
Model note

This AI model was trained on the CIFAR-10 dataset.
It can classify images into 10 predefined categories.

For images outside these categories, predictions may be inaccurate.
```

Do not claim that the model can recognize arbitrary real-world objects.

---

# 18. FRONTEND JAVASCRIPT

Use JavaScript to handle:

* File selection
* Drag and drop
* Image preview
* Predict button
* AJAX/fetch request
* Loading state
* Result rendering
* Clear/reset functionality
* Error messages

Prediction should happen without a full-page reload.

Use:

```javascript
fetch('/predict', {
    method: 'POST',
    body: formData
})
```

Handle JSON responses cleanly.

---

# 19. ERROR UX

Create beautiful user-friendly error states.

Examples:

No file:

```text
Please select an image first.
```

Unsupported format:

```text
Unsupported image format.
Please upload JPG, PNG, or WEBP.
```

Invalid image:

```text
We couldn't read this image.
Please try another file.
```

Prediction error:

```text
Something went wrong while analyzing the image.
Please try again.
```

Do not display technical Python errors to the user.

---

# 20. RESPONSIVE DESIGN

The application must work well on:

### Desktop

Two-column layout:

```text
┌────────────────────┬────────────────────┐
│                    │                    │
│   Upload/Preview   │    Prediction      │
│                    │                    │
└────────────────────┴────────────────────┘
```

### Mobile

Stack vertically:

```text
Upload
   ↓
Preview
   ↓
Predict
   ↓
Prediction
   ↓
Top Predictions
```

Buttons should be large enough for touch interaction.

---

# 21. ANIMATIONS

Use subtle animations only.

Include:

* Fade-in
* Slide-up
* Upload hover animation
* Drag-over animation
* Button hover
* Prediction result reveal
* Confidence bar animation
* Loading spinner

Avoid excessive animations that make the application feel childish.

---

# 22. ACCESSIBILITY

Follow good accessibility practices.

Include:

* Semantic HTML
* Proper labels
* Alt text
* Keyboard-friendly controls
* Visible focus states
* Good color contrast
* Accessible error messages
* Buttons that clearly describe their actions

Do not make important information dependent only on color.

---

# 23. SECURITY

Implement basic Flask upload security.

Use:

```python
secure_filename
```

Validate extensions.

Set maximum upload size.

Never execute uploaded files.

Do not expose uploaded files unnecessarily.

Avoid path traversal vulnerabilities.

Do not include secret API keys or credentials in source code.

---

# 24. MODEL LOADING

The model should be loaded once.

Do NOT do this inside every `/predict` request:

```python
model = load_model(...)
```

Instead, load it when Flask initializes.

The prediction endpoint should reuse the already-loaded model.

If TensorFlow causes unnecessary console logs, configure logging appropriately without hiding real errors.

---

# 25. PERFORMANCE

Keep the application lightweight.

Do not retrain the model when Flask starts.

Do not download CIFAR-10 every time the server starts.

Do not perform unnecessary preprocessing.

Use inference mode appropriately for the installed TensorFlow/Keras version.

If practical, warm up the model once after loading to reduce the delay on the first prediction.

---

# 26. REQUIREMENTS.TXT

Generate an appropriate `requirements.txt`.

Include the packages actually used, such as:

```text
Flask
tensorflow
numpy
Pillow
```

Use compatible versions based on the environment rather than blindly pinning incompatible versions.

---

# 27. README

Create a complete beginner-friendly README.

Include:

## Project Overview

Explain what the application does.

## Model Architecture

Show:

```text
Input Image
    ↓
32 × 32 × 3
    ↓
Flatten
    ↓
3072
    ↓
Dense 312 + ReLU
    ↓
Dense 256 + ReLU
    ↓
Dense 128 + ReLU
    ↓
Dense 10 + Softmax
    ↓
CIFAR-10 Prediction
```

## Installation

Example:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then explain how to open the local Flask URL.

## Model Placement

Explain where:

```text
cifar10_model.keras
```

should be located.

## Prediction Flow

Explain:

```text
Upload
  ↓
RGB conversion
  ↓
Resize to 32×32
  ↓
Normalize /255
  ↓
Add batch dimension
  ↓
Keras model
  ↓
Softmax probabilities
  ↓
Top prediction
```

---

# 28. DO NOT CHANGE THE MODEL ARCHITECTURE

Very important:

Do not modify the trained architecture simply to make the web app work.

The web application must adapt to the existing model.

The model expects:

```text
32 × 32 × 3
```

and produces:

```text
10 probabilities
```

The class ordering must remain exactly:

```python
[
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
```

Therefore:

```text
index 0 → airplane
index 1 → automobile
index 2 → bird
index 3 → cat
index 4 → deer
index 5 → dog
index 6 → frog
index 7 → horse
index 8 → ship
index 9 → truck
```

Do not reorder these classes.

---

# 29. IMPORTANT IMAGE PREPROCESSING DETAIL

The model was trained on images normalized using:

```python
x_train = x_train / 255.0
```

Therefore inference must use exactly the same normalization:

```python
image_array = np.array(image).astype("float32") / 255.0
```

Then:

```python
image_array = np.expand_dims(image_array, axis=0)
```

Expected shape:

```text
(1, 32, 32, 3)
```

Make this preprocessing function separate and easy to understand, for example:

```python
def preprocess_image(image):
    ...
    return image_array
```

---

# 30. CODE QUALITY

Write clean, readable, maintainable code.

Use functions rather than putting everything inside one Flask route.

For example:

```python
load_model()
preprocess_image()
predict_image()
get_top_predictions()
```

Use meaningful variable names.

Add comments around important ML preprocessing logic.

Do not generate unnecessary files.

Do not use hardcoded absolute Windows paths.

Use relative paths based on the Flask application's directory.

---

# 31. FINAL UI EXPERIENCE

The final user flow should feel like this:

```text
                 CIFAR VISION
             See what AI sees.

       AI Image Classification
                 ↓

        ┌──────────────────────┐
        │                      │
        │    Upload Image      │
        │                      │
        │   Drag & Drop Here   │
        │         or           │
        │     Browse Files     │
        │                      │
        └──────────────────────┘

                 ↓

          [ Image Preview ]

                 ↓

       [ ✨ Predict Image ]

                 ↓

          AI is analyzing...

                 ↓

       ┌──────────────────────┐
       │     PREDICTION       │
       │                      │
       │        🐱            │
       │                      │
       │         CAT          │
       │                      │
       │       87.42%         │
       │                      │
       │ ████████████████░░   │
       └──────────────────────┘

                 ↓

          Top Predictions

       Cat       87.42%
       Dog        8.13%
       Frog       2.21%

                 ↓

              [ Clear ]
```

---

# 32. DEVELOPMENT WORKFLOW

Before writing the final code:

1. Inspect the current workspace.
2. Identify whether the trained CIFAR-10 model file exists.
3. Identify the Python/TensorFlow/Keras environment.
4. Reuse the existing trained model instead of retraining.
5. Create the Flask backend.
6. Create the HTML interface.
7. Create CSS styling.
8. Create JavaScript interaction.
9. Connect `/predict` to the frontend.
10. Test image preprocessing.
11. Test prediction.
12. Test clear/reset.
13. Test invalid uploads.
14. Test multiple predictions sequentially.
15. Fix any errors.
16. Make sure the application can be started with one command.

---

# 33. TESTING REQUIREMENTS

Test at least:

### Test 1

Upload a valid JPG.

Expected:

```text
Image preview → Predict → Prediction result
```

### Test 2

Upload PNG.

Expected:

```text
Prediction works.
```

### Test 3

Click Predict without an image.

Expected:

```text
Friendly validation message.
```

### Test 4

Click Clear.

Expected:

```text
Everything resets.
```

### Test 5

Upload image A, predict, then upload image B.

Expected:

```text
Image A prediction
       ↓
Clear/replace
       ↓
Image B prediction
```

### Test 6

Upload unsupported file.

Expected:

```text
Friendly error message.
```

---

# 34. IMPORTANT: VERIFY THE MODEL BEFORE FINALIZING

Do not assume the model file, TensorFlow version, or Keras version.

Inspect the actual workspace.

If a model file exists, verify that it can be loaded.

Verify the model input shape and output shape.

Expected:

```text
Input:  (None, 32, 32, 3)
Output: (None, 10)
```

If the actual saved model differs from the description above, prioritize the **actual saved model** for inference and clearly explain the discrepancy.

Do not silently alter the model.

---

# 35. FINAL DELIVERABLE

At the end, provide:

1. Complete working Flask application.
2. Complete frontend.
3. CSS.
4. JavaScript.
5. Model integration.
6. `requirements.txt`.
7. `README.md`.
8. Proper project structure.
9. Clear instructions to run it.
10. Any model-saving instructions if the model has not already been saved.

The final result should be a polished **CIFAR-10 AI Image Classification web application**, not merely a basic HTML upload form.

Prioritize:

**Correct ML preprocessing → Correct model inference → Clean Flask architecture → Beautiful UX → Responsive design → Error handling → Easy setup.**
