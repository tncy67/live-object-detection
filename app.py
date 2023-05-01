import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
import sqlite3
import os
import time
import pyaudio
import wave
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from email.mime.audio import MIMEAudio
from email import encoders
import threading

def send_email(subject, message, obj_name):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg['From'] = "from@mail.com"
    msg['To'] = "to@mail.com"

    # Attach the text message
    msg.attach(MIMEText(f"{message}\n\nDetected Object: {obj_name}"))

    # Find the latest audio file
    audio_files = [f for f in os.listdir() if f.endswith(".wav")]
    latest_audio_file = max(audio_files, key=os.path.getctime)

    # Attach the latest audio file
    with open(latest_audio_file, "rb") as f:
        audio = MIMEAudio(f.read(), _subtype="wav")
        audio.add_header("Content-Disposition", "attachment", filename=latest_audio_file)
        msg.attach(audio)

    # Find the latest video file
    video_files = [f for f in os.listdir() if f.endswith(".avi")]
    latest_video_file = max(video_files, key=os.path.getctime)

    # Attach the latest video file
    with open(latest_video_file, "rb") as f:
        video = MIMEBase('application', "octet-stream")
        video.set_payload(f.read())
        encoders.encode_base64(video)
        video.add_header("Content-Disposition", "attachment", filename=latest_video_file)
        msg.attach(video)

    # Replace the following line with your SMTP server configuration
    with smtplib.SMTP_SSL('smtp.mail.gmail.com', 465) as server: # modify accordingly for your email provider
        server.login("from@mail.com", "password") # for password, you may need to do "Allow apps that use less secure sign in" or create a target app for your email. It is usually under "Account Security", then "Other ways to sign in"
        server.send_message(msg)

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") 
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

# Detect objects in frame
def detect_objects(img, net, output_layers, classes):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = [(classes[class_ids[i]], confidences[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]) for i in np.array(indexes).ravel()]

    return detected_objects

def create_and_update_database(obj_name, motion_detected, detected_objects):
    conn = sqlite3.connect('objects.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS objects
                 (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  object_name TEXT, 
                  in_motion INTEGER, 
                  other_objects TEXT)''')
    c.execute("INSERT INTO objects (object_name, in_motion, other_objects) VALUES (?, ?, ?)",
              (obj_name, motion_detected, detected_objects))
    conn.commit()
    conn.close()

def is_object_in_motion(current_objects, previous_objects):
    if not previous_objects:
        return False
    for obj in current_objects:
        for prev_obj in previous_objects:
            if obj[0] == prev_obj[0]:  # Compare the class labels
                x_diff = abs(obj[2] - prev_obj[2])
                y_diff = abs(obj[3] - prev_obj[3])
                if x_diff > 20 or y_diff > 20:  # Adjust the threshold as needed
                    return True
    return False

def save_video_clip(filename, cap, duration=5):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        out.write(frame)

    out.release()

def start_audio_recording():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio_filename = f"audio_{int(time.time())}.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    def record_audio():
        frames = []
        for _ in range(0, int(RATE / CHUNK * 5)):
            data = stream.read(CHUNK)
            frames.append(data)

        wf = wave.open(audio_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    audio_recorder = threading.Thread(target=record_audio)
    audio_recorder.start()

    return audio_filename, audio_recorder

def stop_audio_recording(audio_recorder):
    audio_recorder.join()

def start_video_recording(cap, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f"video_{int(time.time())}.avi"
    video_recorder = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
    return video_filename, video_recorder

def stop_video_recording(video_recorder):
    video_recorder.release()

def main():
    global recording
    recording = False

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net, layer_names, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    prev_objects_detected = []

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape

        objects_detected = detect_objects(frame, net, output_layers, classes)
        motion_detected = is_object_in_motion(objects_detected, prev_objects_detected)
        prev_objects_detected = objects_detected

        for obj in objects_detected:
            label, confidence, x, y, w, h = obj
            color = colors[classes.index(label)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.0%}", (x, y - 5), font, 1, color, 1)

        if motion_detected and not recording:
            print("Object in motion detected")
            obj_name = objects_detected[0][0]  # Get the name of the first detected object

            # Start recording audio and video
            audio_filename, audio_recorder = start_audio_recording()
            video_filename, video_recorder = start_video_recording(cap, width, height)
            recording = True
            start_time = time.time()

        if recording and time.time() - start_time > 5:
            # Stop recording audio and video after 5 seconds
            stop_audio_recording(audio_recorder)
            stop_video_recording(video_recorder)
            recording = False

            # Send email and update the database
            send_email("Motion Detected", f"An object ({obj_name}) is in motion!", obj_name)
            detected_objects = ",".join([obj[0] for obj in objects_detected])
            create_and_update_database(obj_name, motion_detected, detected_objects)

        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
