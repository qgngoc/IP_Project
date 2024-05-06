import cv2

def preprocess(image):
    """
    :param image: 3 channel RGB image
    :return: 3 channel RGB image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #thresholded = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)[1]
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    RGB_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return RGB_edges