import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import paddleocr
import cv2
import numpy as np
from paddleocr import PaddleOCR
from matplotlib import pyplot as plt


class SimpleOCR():
    def __init__(self):
        """
        OCR inference scripts based on paddleocr pre-trained models
        requirements:
            python: recommend use python 3.11
            paddlepaddle
                If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
                --> python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
                If you have no available GPU on your machine, please run the following command to install the CPU version
                --> python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
            paddleocr
                pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
            opencv (cv2)
                pip install opencv-python
        """
        self.configurations()
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang,
                 show_log=False, det_db_thresh=self.det_db_thresh,
                 det_db_box_thresh=self.det_db_box_thresh)

    def configurations(self):
        self.upper = 201
        self.lower = 5
        self.ksize = (11, 11)
        self.det_db_box_thresh = 0.5
        self.det_db_thresh = 0.5
        self.lang = 'ch'
        self.sigma = 1.5


    def image_thresholding(self, origin_brg):
        origin_brg = cv2.GaussianBlur(origin_brg, self.ksize, self.sigma)
        gray = cv2.cvtColor(origin_brg.copy(), cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray,
                                     255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,
                                     self.upper,
                                     self.lower)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        image = np.where(mask_rgb == [0, 0, 0], origin_brg, 255)
        return image

    def run_on_image(self, path_to_image: str):
        isClosed = True
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2

        image = cv2.imread(path_to_image)
        image = self.image_thresholding(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(image, cls=True)

        for res in result:
            pt0 = [int(res[0][0][0]), int(res[0][0][1])]
            pt1 = [int(res[0][1][0]), int(res[0][1][1])]
            pt2 = [int(res[0][2][0]), int(res[0][2][1])]
            pt3 = [int(res[0][3][0]), int(res[0][3][1])]

            # Polygon corner points coordinates
            pts = np.array([pt0, pt1, pt2, pt3], np.int32)
            pts = pts.reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts],
                                  isClosed, color, thickness)

        return result, image

    def run_on_folder(self, path_to_folder, is_visualize=True):
        images = os.listdir(path_to_folder)
        for i in images:
            if i[-4:] not in ['.jpg', '.png']:
                continue
            to_image = os.path.join(path_to_folder, i)
            result, image = self.run_on_image(to_image)

            if is_visualize:
                plt.imshow(image)
                plt.show()
                plt.close()

if __name__ == '__main__':
    processor = SimpleOCR()
    #path to image here
    ipath = ''
    #path to image folder here
    #%dir/*.(jpg, png)
    fpath = './ocr'
    #if you wanna run on an image folder
    processor.run_on_folder(fpath)
    #if you wanna run on an image
    processor.run_on_image(ipath)