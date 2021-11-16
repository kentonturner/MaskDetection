import cv2


#Sets up readers for nose, face and mouth
cascade_face = cv2.CascadeClassifier('/usr/local/Cellar/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml') 
cascade_nose = cv2.CascadeClassifier('/usr/local/Cellar/opencv/opencv/data/OpenCV-detection-models/haarcascades/haarcascade_mcs_nose.xml') 
cascade_mouth = cv2.CascadeClassifier('/usr/local/Cellar/opencv/opencv/data/OpenCV-detection-models/haarcascades/haarcascade_mcs_mouth.xml')


def scan(grayscale, img):

    
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    face1 = 0
    face2 = 0
    noseNum = 0
    #For thh face, read values for when a face is found and create a purple rectangle around the face
    for (x_face, y_face, w_face, h_face) in face:
        face1 = x_face
        face2 = y_face
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 130), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        nose = cascade_nose.detectMultiScale(ri_grayscale, 1.2, 18)
        #Reads values for the nose
        for (x_nose, y_nose, w_nose, h_nose) in nose:
            noseNum=h_nose
        mouth = cascade_mouth.detectMultiScale(ri_grayscale, 1.7, 20)
        mouth_num = 0
        for (x_mouth, y_mouth, w_mouth, h_mouth) in mouth:
            num = h_mouth
        #If the mouth and nose are covered, the mask is user is wearing a correctly applied facemask,
        #Output a mask label and change the face identifying rectangle to green
        if(num <5 and noseNum< 5):
            cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (0, 180, 60), 2)
            
            cv2.putText(img, "MASK", (face1, face2), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    return img 
vc = cv2.VideoCapture(0) 
while 1:
    _, img = vc.read() 
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    final = scan(grayscale, img) 
    cv2.imshow('Video', final) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
vc.release() 
cv2.destroyAllWindows() 