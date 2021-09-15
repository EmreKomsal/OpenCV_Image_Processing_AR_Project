import cv2  # openCV python package
import numpy as np  # numpy package

print("Hello OpenCV")


# Function designed for camera capture as image for processing
# optional mirroring for future possible use
# webcamID is computer or system with multiple webcams

def get_webcam(mirror: bool = False, webcamID: int = 0):
    # Use camera device with specified ID 0 for first connected device
    cam = cv2.VideoCapture(webcamID)
    while True:
        # isCameraWorking for camera debug
        # img is frame captured from video camera
        ret_val, img = cam.read()
        # Throw error if couldn't get video stream
        if not ret_val:
            raise NotImplementedError("Webcam couldn't captured")
        # if mirroring the image required mirror it before showing
        if mirror:
            img = cv2.flip(img, 1)
        # show video cam stream as a frame by frame image
        cv2.imshow("Webcam", img)
        # if ESC key pressed end video capture from webcam
        if cv2.waitKey(1) & 0xFF == 27:
            break
    #destroy all video cam windows and free some memory space
    cv2.destroyAllWindows()


def main():
    get_webcam()


if __name__ == '__main__':
    main()
