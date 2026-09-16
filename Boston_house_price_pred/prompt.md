Build a clean, modern, responsive web application for my machine-learning model.

### Project

Create an **AI Prediction Web App** using my trained model file.

### Requirements

* Use **React + Vite + TypeScript**
* Use **Tailwind CSS** for a modern UI
* Create a simple, attractive dashboard
* Add a navigation bar with:

  * Home
  * Prediction
  * About Model

### Prediction Page

Create an easy-to-use prediction interface.

The user should be able to:

1. Enter the required input values.
2. Click **Predict**.
3. Send the values to the backend.
4. Display the prediction in a large result card.

Show:

* Prediction result
* Loading animation
* Error message if prediction fails
* Reset button

### Backend

Use **Python + FastAPI + TensorFlow/Keras**.

Load my trained `.h5` model once when the server starts.

Create:

`POST /predict`

The API should:

1. Receive the input values.
2. Validate them.
3. Apply the required preprocessing.
4. Run the TensorFlow model.
5. Return the prediction.

Also create:

`GET /health`

to check whether the API and model are working.

### UI Design

Make the UI look professional but simple.

Use:

* modern cards
* rounded corners
* clean typography
* subtle shadows
* responsive layout
* attractive buttons
* dark/light mode
* simple animations

Do NOT create unnecessary pages or complicated features.

### Project Structure

Use:

```text
project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── package.json
│
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── requirements.txt
│   └── model/
│       └── trained_model.h5
│
└── README.md
```

### Important

First inspect my `.h5` model and determine:

* input shape
* number of inputs
* output shape
* model architecture
* preprocessing requirements

Do not guess these values.

Build the complete working application and connect the frontend to the FastAPI backend.

Keep the implementation **simple, fast, clean, and easy to run locally**.

Provide the commands required to start both frontend and backend.
