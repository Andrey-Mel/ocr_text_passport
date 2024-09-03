from scanDoc import conf

# import utils
from scanDoc.utils_prj import (
    preproc_im,
    get_class,
    correct_photo,
    load_model,
    get_recog_model,
    prep_img,
    row_fpp,
    ctc_decode,
    num_to_char,
    invert2text_out,
    decode_batch_prediction,
    correct_page,
    output_res,
)

from scanDoc.utils_func import crft_init

import numpy as np

# import pandas as pd
# from PIL import Image
# import cv2
import os

# import matplotlib.pyplot as plt

import tensorflow as tf

# import keras


# from skimage.transform import resize
# from skimage.color import rgb2gray

import ultralytics
from ultralytics import YOLO

ultralytics.checks

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

# recog models
# path_first_model = r"C:\inetpub\wwwroot\FlaskApp\scanDoc\models\recog_first_pg_yolo_v8\weights\best.pt"
recog_model = get_recog_model(conf.path_first_model)  # conf.
model_xcept = load_model(conf.path_turn_model)
print("model row ffp")
model_pr_yolo = YOLO(conf.path_recog_row_model)
print("Ok!")
print("Segment model craft")
crft_model = crft_init()
print("Ok")
print("Model recognition text: ")
# recog_txt_model = tf.keras.models.load_model('model_recog_txt_73', compile=False)
recog_txt_model = tf.keras.models.load_model(conf.path_crnn_model, compile=False)
# type(recog_txt_model)
isinstance(recog_txt_model, type(recog_txt_model))
print("Ok")


# def predict(path_img):
#     path_img = os.path.normpath(path_img)
#     print('PATH IMG: ', path_img)
#     try:
#         print('FUCTION CORRECT PAGE')
#         image = correct_page(
#             path_img, recog_model=recog_model, model_xcept=model_xcept
#         )  # function
#         print('Image shape', image.size())
#         if image is not None:
#             print("ok Image not None!")
#             res = row_fpp(image, model_pr_yolo, crft_model)  # function
#             # print('RES -', res)
#
#         else:
#             raise TypeError
#
#         if res is not None:
#             resHtml = ""
#             print('res not None!!!')
#             print("Получаю результаты...")
#
#             out_res, prob_res = output_res(res, recog_txt_model)
#             # print("Result - ", out_res)
#
#             for k, v in out_res.items():
#                 resHtml += f"{k},{str(out_res[k])}, {str(prob_res[k])};"
#
#             if isinstance(resHtml, str):
#                 print("Ready")
#                 return resHtml
#                 # return render_template("index.html", prediction_text=resHtml)
#
#         else:
#             raise Exception
#
#     except TypeError as t:
#         return "Попробуйте другой скан паспорта"
#
#
#     except Exception as tt:
#
#         return "Невыявлены значения строк, res - None,  попробуйте другой скан!!!"


def predict(path_img):
    path_img = os.path.normpath(path_img)
    print("PATH IMG: ", path_img)

    print("FUCTION CORRECT PAGE")
    image = correct_page(
        path_img, recog_model=recog_model, model_xcept=model_xcept
    )  # function

    if image is not None:
        print("ok Image not None!")
        res = row_fpp(image, model_pr_yolo, crft_model)  # function
        # print('RES -', res)

    else:
        print("image None", image.shape)

    if res is not None:
        resHtml = ""
        print("res not None!!!")
        print("Получаю результаты...")

        out_res, prob_res = output_res(res, recog_txt_model)
        # print("Result - ", out_res)

        for k, v in out_res.items():
            resHtml += f"{k},{str(out_res[k])}, {str(prob_res[k])};"

        if isinstance(resHtml, str):
            print("Ready")
            return resHtml
            # return render_template("index.html", prediction_text=resHtml)

    else:
        print("res is none", len(res))
