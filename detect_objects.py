import cv2
from ultralytics import YOLO
from google.colab import files

model = YOLO('yolov8n.pt')

def detect_in_image(image_path='input.jpg', output_path='output_image.jpg'):
    """Detect objects in a single image and save the output."""
    img = cv2.imread(image_path)
    results = model(img)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0].item()
        class_id = int(result.cls[0])
        label = model.names[class_id]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

def detect_in_video(video_path='input_video.mp4', output_path='output_video.mp4', skip_frames=2):
    """Detect objects in a video and save the output."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        results = model(frame)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0].item()
            class_id = int(result.cls[0])
            label = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)  
        frame_count += 1  
    cap.release()
    out.release()

def main():
    """Main function to upload files and call the detection functions."""
    print("Upload an image or video file for object detection.")

    uploaded = files.upload()

    for filename in uploaded.keys():
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            print(f'Processing image: {filename}')
            detect_in_image(filename, 'output_image.jpg')
            print("Detection complete. Downloading output image...")
            files.download('output_image.jpg')
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            print(f'Processing video: {filename}')
            detect_in_video(filename, 'output_video.mp4', skip_frames=2)  
            print("Detection complete. Downloading output video...")
            files.download('output_video.mp4')
        else:
            print("Unsupported file format. Please upload an image or video.")

main()
