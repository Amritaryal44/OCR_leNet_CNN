import pickle
import numpy as np
import cv2

###############################
width = 640
height = 480
threshold = 0.65
###############################

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32,32))
    img = preprocessing(img)
    img = img.reshape(1,32,32,1)
    #predict
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)

    if probVal>threshold:
        cv2.putText(imgOriginal, str(classIndex)+" "+str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv2.imshow("Original Image", imgOriginal)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
