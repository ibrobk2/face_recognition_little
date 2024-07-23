import cv2 as cv
import numpy as np
import face_recognition as fr

cap = cv.VideoCapture(0)
image = fr.load_image_file('ibrobk1.png')
image_encoding = fr.face_encodings(image)[0]
known_face = [image_encoding]
known_face_name = ["Ibrahim Bakori"]


while True:
    success, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face, face_encoding)
        
        name = "Unknown Face"
        
        face_distances = fr.face_distance(known_face, face_encoding)
        
        match_index = np.argmin(face_distances)
        
        if matches[match_index]:
            name = known_face_name[match_index]
        else:
            name = "Unlnown"
            
        cv.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        cv.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv.FILLED)
        
        font = cv.FONT_HERSHEY_SIMPLEX
        
        cv.putText(frame, name, (left +6, bottom -6), font, 1.0, (255,255,255), 1)
    cv.imshow('FACE RECOGNITION SYSTEM', frame)
    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()
    