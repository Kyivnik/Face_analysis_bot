import cv2
import time
from telebot import TeleBot

bot = TeleBot("TELEGRAM_BOT_TOKEN")

face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
age_gender_net = cv2.dnn.readNetFromCaffe("path/to/deploy_age_gender.prototxt", "path/to/age_gender.caffemodel")
emotion_net = cv2.dnn.readNetFromCaffe("path/to/deploy_emotion.prototxt", "path/to/emotion.caffemodel")

age_list = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_list = ["Female", "Male"]
emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@bot.message_handler(commands=['analyze_face'])
def analyze_face(message):
    
    file_id = message.photo[-1].file_id
    file = bot.get_file(file_id)
    file.download("face.jpg")

    image = cv2.imread("face.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))

        age_gender_net.setInput(cv2.dnn.blobFromImage(face_roi, 1, (48, 48), (104, 117, 123), swapRB=False))
        pred_age_gender = age_gender_net.forward()

        age = age_list[pred_age_gender[0].argmax()]
        gender = gender_list[pred_age_gender[1].argmax()]

        emotion_net.setInput(cv2.dnn.blobFromImage(face_roi, 1, (48, 48), (104, 117, 123), swapRB=False))
        pred_emotion = emotion_net.forward()

        emotion = emotion_list[pred_emotion.argmax()]

        cv2.putText(image, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.putText(image, f"Gender: {gender}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.putText(image, f"Emotion: {emotion}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    cv2.imwrite("face_analyzed.jpg", image)
    with open("face_analyzed.jpg", "rb") as f:
        bot.send_photo(chat_id=message.chat.id, photo=f)
      
bot.polling(none_stop=True, interval=0)
