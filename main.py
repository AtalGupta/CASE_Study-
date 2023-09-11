from collections import defaultdict
import cv2

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

input_video_path = "output.mp4"
cap = cv2.VideoCapture(input_video_path)

roi = (130, 110, 300, 400)  # (x, y, w, h)

track_history = defaultdict(lambda: [])

output_videos = {}

# Loop through the input video frames
while cap.isOpened():

    success, frame = cap.read()
    if success:

        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        results = model.track(frame, persist=True)

        for result in results:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            annotated_frame = result.plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            # Check if this track intersects with the ROI
            if roi[0] < x < roi[0] + roi[2] and roi[1] < y < roi[1] + roi[3]:
                # Check if we have already created an output video for this track ID
                if track_id not in output_videos:
                    # Create a new VideoWriter object for this track ID
                    output_video_path = f"output_video_{track_id}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (roi[2], roi[3]))

                    output_videos[track_id] = out

                # Save this frame to the output video for this track ID
                roi_frame = annotated_frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                output_videos[track_id].write(roi_frame)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # break when end is reached
        break

# Release all window
for out in output_videos.values():
    out.release()
cap.release()
cv2.destroyAllWindows()
