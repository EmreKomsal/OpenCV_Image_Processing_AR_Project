import cv2  # openCV python package
import numpy as np  # numpy package


# Function designed for camera capture as image for processing
# optional mirroring for future possible use
# webcamID is computer or system with multiple webcams

def get_webcam(mirror: bool = False, webcamID: int = 0):
    # Use camera device with specified ID 0 for first connected device
    # CAP_DSHOW added because of the windows based error
    cam = cv2.VideoCapture(webcamID, cv2.CAP_DSHOW)
    while True:
        minValue = 20
        maxValue = 50
        # isCameraWorking for camera debug
        # img is frame captured from video camera
        ret_val, img = cam.read()
        # Throw error if couldn't get video stream
        if not ret_val:
            raise NotImplementedError("Webcam couldn't captured")
        # if mirroring the image required mirror it before showing
        if mirror:
            img = cv2.flip(img, 1)
        # Creating gray scale version of image for test
        img_grayScale = GrayScale_Image_3_Channel(img)
        # Create Canny and Blur Filters
        img_blur, img_canny = Blur_and_Canny_Ege_Image(img, 7, (minValue, maxValue))
        # Stacking images in one window
        img_stack = np.hstack((img, img_grayScale))
        img_stack2 = np.hstack((img_blur, img_canny))
        # show video cam stream as a frame by frame image
        cv2.imshow("Webcam", img_stack)
        cv2.imshow("Webcam_Processed", img_stack2)
        # if ESC key pressed end video capture from webcam
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # destroy all video cam windows and free some memory space
    cv2.destroyAllWindows()


# Find most fitting method for video types
def Image_Window_Optimizer(img_list):
    # Find array length for appropriate fit
    array_length = len(img_list)



# Make image 3 channel Grayscale Image
def GrayScale_Image_3_Channel(img):
    # Creating gray scale version of image
    img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Making 2 channel image back to 3 channel image
    img_grayScale_3_Channel = cv2.cvtColor(img_grayScale, cv2.COLOR_GRAY2BGR)
    return img_grayScale_3_Channel


# Make blurred and cannied versions of image
# First Canny Paramaters for Max value, second value for Min value
def Blur_and_Canny_Ege_Image(img, matrix_size: int, canny_parameters: (int, int)):
    # Creating gray scale version of image for Gaussian Blur
    img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_matrix = (matrix_size, matrix_size)
    # Applying Gaussuan Blur to image
    img_blur = cv2.GaussianBlur(img_grayScale, kernel_matrix, 0)
    # Finding Canny Edges of image
    img_canny = cv2.Canny(img_blur, canny_parameters[0], canny_parameters[1])
    return img_blur, img_canny


def main():
    get_webcam()


if __name__ == '__main__':
    main()
