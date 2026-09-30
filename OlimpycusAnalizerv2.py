import cv2
import mediapipe as mp
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
draw = mp.solutions.drawing_utils

def angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    ang = np.abs(radians*180/np.pi)

    if ang>180:
        ang=360-ang

    return ang

def analyzeVideo(path):
    cap=cv2.VideoCapture(path)
    elbowAngles = []
    hipAngles = []
    shoulderAngles = []
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            draw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            shoulder=[lm[12].x, lm[12].y]
            elbow=[lm[14].x, lm[14].y]
            wrist=[lm[16].x, lm[16].y]
            hip=[lm[24].x, lm[24].y]
            knee=[lm[26].x, lm[26].y]
            shoulder2=[lm[11].x, lm[11].y]

            elbowAngle = angle(shoulder,elbow,wrist)
            hipAngle = angle(shoulder, hip, knee)
            shoulderAngle = angle(shoulder2, shoulder, hip)

            elbowAngles.append(elbowAngle)
            hipAngles.append(hipAngle)
            shoulderAngles.append(shoulderAngle)
        frames.append(frame)
    
    cap.release()
    return np.mean(elbowAngles), np.mean(hipAngles), np.mean(shoulderAngles),frames

print("Analyzing Tiger Woods' siwng")
tigerElbow, tigerHip, tigerShoulder, tigerFrames = analyzeVideo("tiger_swing.mp4")

print("Tiger elbow: ", tigerElbow)
print("Tiger hip: ", tigerHip)
print("Tiger shoulder: ", tigerShoulder)

print()
uo = input("1)Use the camera   |   2)Upload a video") 

if uo == "1":
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    userElbow = []
    userHip = []
    userShoulder = []
    userFrames = [] 

    swingState = "READY"

    extendedCounter = 0
    bendCounter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading camera")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            draw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark

            shoulder=[lm[12].x, lm[12].y]
            elbow=[lm[14].x, lm[14].y]
            wrist=[lm[16].x, lm[16].y]
            hip=[lm[24].x, lm[24].y]
            knee=[lm[26].x, lm[26].y]
            shoulder2=[lm[11].x, lm[11].y]

            elbowAngle = angle(shoulder,elbow,wrist)
            hipAngle = angle(shoulder, hip, knee)
            shoulderAngle = angle(shoulder2, shoulder, hip)

            cv2.putText(frame,"Elbow:"+str(int(elbowAngle)),(30,50), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            if elbowAngle>165:
                extendedCounter +=1
            else:
                extendedCounter = 0

            if elbowAngle<140:
                bendCounter+=1
            else:
                bendCounter=0

            if swingState =="READY" and bendCounter>5:
                print("Swing Started")
                swingState ="BACKSWING" 

            if swingState!="READY":
                userFrames.append(frame.copy())
                userElbow.append(elbowAngle)
                userHip.append(hipAngle)
                userShoulder.append(shoulderAngle)

            if swingState=="BACKSWING" and extendedCounter>5:
                print("Swing Finished")
                break

        cv2.imshow("Swing Analyzer", frame)

        cv2.waitKey(1)

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows

else:
    path = input("Video's path or Video's name: ")

    userElbow, userHip, userShoulder, userFrames = analyzeVideo(path)

print()
print("RESULTS")

userElbow = np.mean(userElbow)
userHip = np.mean(userHip)
userShoulder = np.mean(userShoulder)

print("Your elbow: ", userElbow)
print("Your hip: ", userHip)
print("Your shoulder: ", userShoulder)

print("Showing visual comparison: ")

tigerAngles=[]
userAngles=[]

for f in tigerFrames:
    img=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    r=pose.process(img)

    if r.pose_landmarks:
        lm=r.pose_landmarks.landmark
        tigerAngles.append(angle([lm[12].x,lm[12].y],[lm[14].x,lm[14].y],[lm[16].x,lm[16].y]))
    else:
        tigerAngles.append(180)

for f in userFrames:
    img=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
    r=pose.process(img)

    if r.pose_landmarks:
        lm=r.pose_landmarks.landmark
        userAngles.append(angle([lm[12].x,lm[12].y],[lm[14].x,lm[14].y],[lm[16].x,lm[16].y]))
    else:
        userAngles.append(180)

tigerImpact=np.argmin(tigerAngles)
userImpact=np.argmin(userAngles)

offset=userImpact-tigerImpact

i=0

while True:
    t = (i) % len(tigerFrames)
    u = (i + offset) % len(userFrames)

    tiger = cv2.resize(tigerFrames[t],(640,480))
    user = cv2.resize(userFrames[u],(640, 480))

    tigerRGB=cv2.cvtColor(tiger,cv2.COLOR_BGR2RGB)
    userRGB=cv2.cvtColor(user,cv2.COLOR_BGR2RGB)

    tigerRes=pose.process(tigerRGB)
    userRes=pose.process(userRGB)

    if tigerRes.pose_landmarks:
        draw.draw_landmarks(tiger, tigerRes.pose_landmarks,mpPose.POSE_CONNECTIONS)

    if userRes.pose_landmarks:
        draw.draw_landmarks(user, userRes.pose_landmarks,mpPose.POSE_CONNECTIONS)

    combined=np.hstack((tiger,user))

    cv2.putText(combined,"Tiger Woods", (200, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.line(combined,(640,0), (640,480),(255,255,255),2)
    cv2.putText(combined, "User", (900,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("Swing Comparison", combined)

    i+=1

    if cv2.waitKey(40)==ord("q"):
        break

cv2.destroyAllWindows()

userTotal = userElbow + userHip + userShoulder
tigerTotal = tigerElbow + tigerHip + tigerShoulder

if abs(userElbow - tigerElbow)>15:
    print("Keep your arm more straight")
if abs(userHip - tigerHip)>15:
    print("Rotate your hips more")
if abs(userShoulder - tigerShoulder)>15:
    print("Rotate your shoulders more")
if abs(userTotal - tigerTotal)<10:
    print("A very similar swing to Tiger Woods!")

