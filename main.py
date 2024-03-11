import cv2

face = cv2.CascadeClassifier("face.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimaze_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face.detectMultiScale(optimaze_frame, scaleFactor=1.1, minSize=(500, 500), minNeighbors=3)
    return faces

def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 4)
    pass

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("face_scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    

if __name__ == "__main__" :
    main()
