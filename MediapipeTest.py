import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()