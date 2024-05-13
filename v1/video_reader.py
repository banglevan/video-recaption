import time
import os
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
# from translator_gpt import openai_translator as zhvi_translator
# from translator_langchain import translator_lc as zhvi_translator
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from PIL import ImageFont, ImageDraw, Image

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_COMPLEX,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    # x, y = pos
    # text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    # text_w, text_h = text_size
    # cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    # cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    b, g, r, a = 255, 255, 255, 255

    ## Use cv2.FONT_HERSHEY_XXX to write English.
    # text = time.strftime("%Y/%m/%d %H:%M:%S %Z", time.localtime())
    # cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (b, g, r), 1, cv2.LINE_AA)

    ## Use simsum.ttc to write Chinese.
    fontpath = "data/arial.ttf"  # <== 这里是宋体路径
    font = ImageFont.truetype(fontpath, 35)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((200, 200), text, font=font, fill=(b, g, r, a))
    img = np.array(img_pil)

    return img

class VideoReader():
    def __init__(self):
        self.recognizer_initialize()
        self.logo_image = cv2.imread('data/logo.png')
        self.ratio = 0.22
        self.split_point = 0.55
        self.minimal_area = 125
        self.frame_width = 0#int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = 0#int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def merge_text(self, rects):
        ptxs = []
        txts = []
        for r in rects:
            ptx = r[0][0][1]
            ptxs.append(ptx)
            txts.append(r[1])
        txts = np.array(txts)
        idxs = np.argsort(ptxs)
        stxts = txts[idxs]
        res = "".join(x for x in stxts)
        return res.strip()

    def add_noise(self, image):
        # Generate random Gaussian noise
        mean = 0
        stddev = 5
        noise = np.zeros(image.shape, np.uint8)
        cv2.randn(noise, mean, stddev)

        # Add noise to image
        noisy_img = cv2.add(image, noise)
        return noisy_img

    def recognizer_initialize(self):
        # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
        # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch",
                             show_log=False, det_db_thresh=0.95,
                             det_db_box_thresh=0.8)
        # need to run only once to download and load model into memory

    def frame_text_recognition(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(image, cls=True)
        rects = []
        masks = []
        for idx in range(len(result)):
            res = result[idx]
            pt0 = (int(res[0][0][0]), int(res[0][0][1]))
            pt2 = (int(res[0][2][0]), int(res[0][2][1]))
            cond1 = (pt2[0] - pt0[0]) * (pt2[1] - pt0[1]) > self.minimal_area
            cond2 = pt0[1] > self.filter_loc_upper
            cond3 = pt0[1] < self.filter_loc_lower
            if cond3:
                masks.append((pt0, pt2))

            if cond1 and cond2 and cond3:
                r = [(pt0, pt2), res[1][0]]
                rects.append(r)
                # print(r)
        return rects, masks

    def mask_bbox(self, pts, image):
        if len(pts) > 0:
            for (pt0, pt2) in pts:
                arr = image[pt0[1]:pt2[1], pt0[0]:pt2[0], :]
                if arr.shape[0] * arr.shape[1] != 0:
                    blur = cv2.GaussianBlur(arr, (15, 15), 5.85)
                    image[pt0[1]:pt2[1], pt0[0]:pt2[0], :] = blur
        return image

    def blending_by_bbox(self, fpt0, fpt2, image, tar=None):
        pt0 = (int(fpt0[0]), int(fpt0[1]))
        pt2 = (int(fpt2[0]), int(fpt2[1]))
        alpha = .5
        arr = image[pt0[1]:pt2[1], pt0[0]:pt2[0], :]
        if tar is None:
            tar = np.zeros_like(arr)# + 255

        dst = cv2.addWeighted(arr, alpha, tar, 1 - alpha, 0.0)
        blur = cv2.GaussianBlur(dst, (15, 15), 3.85)
        image[pt0[1]:pt2[1], pt0[0]:pt2[0], :] = blur

        return image

    def get_frame(self, path:str):
        # Create a VideoCapture object and read from input file

        rfpt0 = [240.0, 740.0]
        rfpt2 = [472.0, 762.0]

        lfpt0 = [14.0, 742.0]
        lfpt2 = [152.0, 755.0]

        cap = cv2.VideoCapture(path)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video file")

        prev_txt = ''
        curr_txt = ''

        prev_trans = ''
        curr_trans = ''
        curr_trans_ = ''
        stt = ''
        n_caps = 0
        n_frame = 0
        start_st = ''
        end_st = ''
        last_st = ''

        infors = []

        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # grab the width, height, and fps of the frames in the video stream.
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                break

        cap_stt = False
        # initialize the FourCC and a video writer object
        fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
        output = cv2.VideoWriter('results/output.mp4', fourcc, self.fps, (self.frame_width, self.frame_height))
        # Read until video is completed
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                tic = time.time()
                n_frame += 1
                located = n_frame / self.fps
                convert = time.strftime("%H:%M:%S", time.gmtime(located))
                h, m, s = convert.strip().split(':')
                ms = (located - (float(h) * 60 * 60 + float(m) * 60 + float(s))) * 1000
                timestamp = convert + ',' + f'{int(ms):03d}'

                # Display the resulting frame
                self.filter_loc_lower = int(max(frame.shape[:2]) * (1 - self.ratio))
                self.filter_loc_upper = int(max(frame.shape[:2]) * self.split_point)

                minimal_size = float(min(frame.shape[:2]) * self.ratio) / self.logo_image.shape[0]

                logo_image = cv2.resize(self.logo_image, (0, 0), fx=minimal_size, fy=minimal_size)

                lpt0 = [0, frame.shape[0] - logo_image.shape[0]]
                lpt2 = [logo_image.shape[0], frame.shape[0]]

                frame = self.blending_by_bbox(lpt0, lpt2, frame, logo_image)
                frame = self.blending_by_bbox(rfpt0, rfpt2, frame)
                frame = self.blending_by_bbox(lfpt0, lfpt2, frame)
                rects, masks = self.frame_text_recognition(frame)

                if len(rects) > 0:
                    curr_txt = self.merge_text(rects)
                    if fuzz.ratio(curr_txt, prev_txt) < 90:
                        n_caps += 1
                        start_st = timestamp
                        # curr_trans = zhvi_translator(curr_txt)
                        prev_trans = curr_trans
                        prev_txt = curr_txt
                        stt = 'process'
                        # end_sts = last_st

                    else:
                        curr_trans = prev_trans
                        stt = 'iher'
                        last_st = timestamp
                    infors.append([n_caps, start_st, last_st, curr_txt])
                    # todo: run drawing
                    # frame = draw_text(frame, curr_trans)
                    print(len(infors), n_caps, start_st, last_st, curr_txt, curr_trans)
                frame = self.mask_bbox(masks, frame)
                # for r in rects:
                #     cv2.rectangle(frame, r[0][0], r[0][1], (0, 0, 255), 2)
                output.write(frame)
                # frame = self.add_noise(frame)
                toc = time.time()
                # print(f'{toc - tic:.4f}', curr_txt, curr_trans, stt)
                frame = cv2.resize(frame, (0, 0), fx=.75, fy=.75)
                cv2.imshow('Frame', frame)
                # time.sleep(0.5)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        if len(infors) > 0:
            df = pd.DataFrame(infors, columns=['n_cap', 'stime', 'etime', 'content'])
            df.to_csv('results/infors.csv', encoding='utf_8_sig', index=False)
            print('----------------------')
            idxs = [i[0] for i in infors]
            idxs = list(np.unique(np.array(idxs)))
            print(idxs)
            for c in idxs:
                arr = [i for i in infors if i[0] == c]
                s = arr[0][1]
                o = arr[0][3]
                if len(arr) > 1:
                    e = arr[-1][2]
                    # t = arr[0][4]
                else:
                    e = s
                print(c, s, e, o)

        # When everything done, release
        # the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

if __name__ == '__main__':
    reader = VideoReader()
    path = r'C:\BANGLV\capcut\ep4\ex4.mp4'
    reader.get_frame(path)