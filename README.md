# Sign Language to Text Converter

A real-time sign language recognition system that converts hand gestures into text using computer vision and machine learning technologies.

## Project Overview

This project was developed to help improve communication accessibility by translating sign language gestures into written text in real time.

The system uses:

- **Python** for backend processing
- **OpenCV** for image processing
- **MediaPipe** for hand landmark detection
- **Flask** for the web interface
- **WebSocket** for real-time communication
- A **custom-trained machine learning model** developed by our team

The application captures hand movements from a camera feed, processes the gesture data, and instantly converts recognized signs into text on the web interface.

---

## Features

- Real-time sign language recognition
- Live camera streaming
- Instant text generation
- Web-based user interface
- Low-latency communication using WebSocket
- Custom-trained recognition model
- Hand tracking with MediaPipe
- Lightweight and fast architecture

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core backend development |
| OpenCV | Image processing |
| MediaPipe | Hand landmark detection |
| Flask | Web application framework |
| WebSocket | Real-time data transfer |
| Machine Learning | Gesture classification |

---

## System Architecture

1. Camera captures hand gestures
2. OpenCV processes video frames
3. MediaPipe extracts hand landmarks
4. The trained model predicts the sign
5. Prediction results are sent through WebSocket
6. Flask interface displays the generated text in real time

---

## How It Works

The system continuously analyzes hand movements from the webcam feed. Hand landmark data extracted by MediaPipe is passed into our trained model. The predicted sign is then converted into text and displayed instantly on the web interface.

The communication between the processing backend and the frontend interface is handled using WebSocket technology to ensure real-time performance.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
python app.py
```

## Future Improvements
1. Sentence prediction support
2. More sign language datasets
3. Multi-language support
4. Mobile compatibility
5. Higher recognition accuracy
6. Speech output support


## Team
Developed collaboratively as a computer vision and accessibility project.

