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
        img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Create Canny and Blur Filters
        img_blur, img_canny = Blur_and_Canny_Ege_Image(img, 7, (minValue, maxValue))
        img_contour = getContours(img_canny, img)
        # Stacking images in one window
        img_list = [[img, img_grayScale], [img_blur, img_canny], [img_contour, img_contour]]
        img_stack = stackImages(0.5, img_list)
        # show video cam stream as a frame by frame image
        cv2.imshow("Webcam", img_stack)
        # if ESC key pressed end video capture from webcam
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # destroy all video cam windows and free some memory space
    cv2.destroyAllWindows()


# Find most fitting method for video types
def stackImages(scale,imgArray):
    # get row length
    rows = len(imgArray)
    # get column length from 2D imgArray
    cols = len(imgArray[0])
    # check if rows is array
    rowsAvailable = isinstance(imgArray[0], list)
    # use camera feed as a image size
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Make blurred and cannied versions of image
# First Canny Parameters for Max value, second value for Min value
def Blur_and_Canny_Ege_Image(img, matrix_size: int, canny_parameters: (int, int)):
    # Creating gray scale version of image for Gaussian Blur
    img_grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_matrix = (matrix_size, matrix_size)
    # Applying Gaussian Blur to image
    img_blur = cv2.GaussianBlur(img_grayScale, kernel_matrix, 0)
    # Finding Canny Edges of image
    img_canny = cv2.Canny(img_blur, canny_parameters[0], canny_parameters[1])
    return img_blur, img_canny


def getContours(img, img_original, areaSize=200):
    img_contour = img_original.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaSize:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
            cornerLength = cv2.arcLength(cnt, True)
            approx_shape = cv2.approxPolyDP(cnt, 0.02 * cornerLength, True)
            origin_x, origin_y, shape_width, shape_height = cv2.boundingRect(approx_shape)
            cv2.rectangle(img_contour, (origin_x, origin_y), (origin_x+shape_width, origin_y+shape_height),
                          (255, 0, 255), 2)
    return img_contour


def main():
    get_webcam(mirror= True)


if __name__ == '__main__':
    main()
