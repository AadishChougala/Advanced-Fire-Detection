from ultralytics import YOLO
import winsound
import cv2

frequency = 500
duration = 1000

# model = YOLO('weapon_15_epoch_best.pt')
model = YOLO('fire.pt')


video_path = "sample_1.mp4"
# video_path = 0
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame, stream_buffer=False, conf=0.6, vid_stride=25, show_labels=True, show_conf=False)
        names = model.names

        for c in results[0].boxes.cls:
            if names[int(c)] == 'fire':
                print("Alert!!\n")
                winsound.Beep(frequency, duration)
                
                break
            
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 640))
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()