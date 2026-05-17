# Mini_Project_2026
# Smart Surveillance System using Face Authentication & Threat Detection

## Overview
This project is an AI-powered **smart surveillance system** that combines **face authentication, object/threat detection, and real-time alerts** to improve security monitoring. The system verifies authorized users through facial recognition and detects potentially dangerous objects using YOLOv8.

When an unknown person or threat is detected, alerts are triggered automatically to enhance safety and monitoring efficiency.

---

## Features

- 🔐 Face authentication using **LBPH Face Recognizer**
- 👤 Authorized and unauthorized user identification
- 🎯 Real-time object detection using **YOLOv8**
- ⚠️ Threat detection for suspicious objects
- 📩 Telegram alert notifications
- 📷 Webcam-based live monitoring
- 🧠 Identity locking mechanism to reduce false detections
- ⏱ Alert cooldown system to prevent repeated notifications
- 💻 User-friendly surveillance interface

---

## Technologies Used

- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- LBPH Face Recognition  
- DeepFace  
- NumPy  
- HTML / CSS  
- Telegram Bot API  

---

## Project Structure

```bash
Mini_Project_2026/
│
├── models/
│   └── lbph_model.xml
│
├── src/
│   ├── main.py
│   ├── face_auth.py
│   ├── object_detector.py
│   ├── train_lbph.py
│   ├── extract_frames.py
│   ├── alert.py
│   ├── ui.py
│   └── ui.html
│
└── README.md
```

---



## Workflow

1. Capture live webcam feed  
2. Authenticate user using facial recognition  
3. Detect objects and potential threats using YOLOv8  
4. Trigger alerts for suspicious activity or unauthorized access  
5. Continue monitoring in real time  

---

## Applications

- Smart home security  
- Campus/hostel surveillance  
- Office access monitoring  
- Restricted area protection  
- Laboratory security systems  

---

## Future Enhancements

- Multiple user authentication  
- Cloud-based data storage  
- Email/SMS notifications  
- Mobile application integration  
- Improved threat categories and detection accuracy  

---

## Contributors

Developed as an academic mini project by:

- Giriraj Pande 
- Janhavi Chede 
- Manjiri Agrawal 
- Chinmay Khare 

---

## License

This project is created for **educational and research purposes**.

---

## Acknowledgements

Special thanks to:

- OpenCV  
- Ultralytics YOLOv8  
- DeepFace  
- Python open-source community  

