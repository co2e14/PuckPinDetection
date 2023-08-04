import cv2
import time
import os

def capture_images(interval=5, output_folder='images'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Capturing images. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Generate a unique filename
        filename = os.path.join(output_folder, f'{int(time.time())}.png')
        
        # Save the image
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        time.sleep(interval)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Capture stopped.")

if __name__ == "__main__":
    capture_images()
