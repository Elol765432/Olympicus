import cv2
import mediapipe as mp
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle

def extractAngles(videoPath):
    cap = cv2.VideoCapture(videoPath)
    angles = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            shoulder = [lm[12].x, lm[12].y]
            elbow = [lm[14].x, lm[14].y]
            wrist = [lm[16].x, lm[16].y] 

            angle = calculateAngle(shoulder, elbow, wrist)
            angles.append(angle)
    
    cap.release()

    if len(angles) == 0:
        return 0
    
    return np.mean(angles)

tigerAngle = extractAngles("tiger_swing.mp4")
print("Angulo promedio de Tiger Woods: ", tigerAngle)

uo = input("1) Usar camara | 2) Subir video")

if uo == "1":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("No se pudo abrir la camara")
        exit()

    userAngles = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error leyendo la camara")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            shoulder = [lm[12].x, lm[12].y]
            elbow = [lm[14].x, lm[14].y]
            wrist = [lm[16].x, lm[16].y]

            angle = calculateAngle(shoulder, elbow, wrist)
            userAngles.append(angle)

            cv2.putText(frame, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Swing Analyzer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if len(userAngles) == 0:
        userAngle = 0
    else:
            userAngle = np.mean(userAngles)
else:
    path = input("Ruta del video: ")
    userAngle = extractAngles(path)

print("Tu angulo promedio es:", userAngle)

difference = abs(userAngle - tigerAngle)
print("Diferencia de angulo:", difference)

if difference < 10:
    print("Swing muy parecido al de Tiger Woods")
elif userAngle < tigerAngle:
    print("Dobla más el brazo durante el swing")
else:
    print("Manten el brazo más recto como Tiger Woods")


cv2.destroyAllWindows()