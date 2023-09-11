import cv2

# Input video file name (replace with your video file name)
input_video = 'video.mp4'

# Output video file name (replace with your desired output file name)
output_video = 'output.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video)


fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_video, fourcc, 1, (frame_width, frame_height))

# Initialize variables
frame_count = 0
frame_interval = fps  # Preserve one frame every second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Preserve only the middle frame at every 1-second interval
    if frame_count % frame_interval == frame_interval // 2:
        out.write(frame)

# Release the video captures and writers
cap.release()
out.release()
cv2.destroyAllWindows()

print(f'Video processed and saved as {output_video}')
