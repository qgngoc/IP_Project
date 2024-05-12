import cv2
class DataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def edge_filtering(image):
        """
        :param image: 3 channel RGB image
        :return: 3 channel RGB image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #thresholded = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)[1]
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        RGB_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return RGB_edges

    @staticmethod
    def data_augmentation(data):
        """
        :param data: (image, label).
        :return: list of augmented data with same format
        """
        
        # Flip (mirror) image and corresponding bounding box
        img, label = data
        flipped_img = cv2.flip(img, 1)
        boxes = label[1]

        for i in range(0, len(boxes)):
            boxes[i][0] = len(img[0]) - boxes[i][0]
            boxes[i][2] = len(img[0]) - boxes[i][2]

        flipped_label = (label[0], boxes)
        data = (flipped_img, flipped_label)


        augmented_data = [data]
        return augmented_data

    # @staticmethod
    # def preprocess(image):
    #     """
    #     :param image: 3 channel RGB image
    #     :return: 3 channel RGB image
    #     """
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     thresholded = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)[1]
    #     edges = cv2.Canny(thresholded, 50, 150, apertureSize=3)
    #     RGB_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    #
    #     return RGB_edges
    @staticmethod
    def read_image(image):
        return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

    @staticmethod
    def apply_ben_preprocessing(image):    
        return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)

    @staticmethod
    def apply_denoising(image):   
        return cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)