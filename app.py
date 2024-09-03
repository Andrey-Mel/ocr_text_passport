import conf

# import utils
from utils_prj import (
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

from utils_func import crft_init

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from flask import Flask, request, flash, redirect, send_file, render_template

from skimage.transform import resize
from skimage.color import rgb2gray

import ultralytics
from ultralytics import YOLO

ultralytics.checks

import warnings

warnings.filterwarnings("ignore")


app = Flask(__name__)

# recog models
recog_model = get_recog_model(conf.path_first_model)
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
type(recog_txt_model)
isinstance(recog_txt_model, type(recog_txt_model))

PATH_TO_IMG = r"C:\ml\Codes\RECOG_DOCS\CREATE_DATA_RCNN__TRAIN\data_first_pg\test"


@app.route("/")  # homepage
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        uploaded_file = request.files.get("file")
        path_img = PATH_TO_IMG + "\\" + uploaded_file.filename
        print(path_img)
        image = correct_page(
            path_img, recog_model=recog_model, model_xcept=model_xcept
        )  # function

        if image is not None:
            print("ok")
            res = row_fpp(image, model_pr_yolo, crft_model)  # function
            # plt.imshow(image)
            # plt.show()
        else:
            raise TypeError

        if res is not None:
            print()
            print("Получаю результаты...")
            print()
            np.random.seed(42)
            out_res, prob_res = output_res(res, recog_txt_model)
            print("Result - ", out_res)

            # prob_list = list(prob_res.values())
            resHtml = ""
            for k, v in out_res.items():
                resHtml += f"{k},{str(out_res[k])}, {str(prob_res[k])};"
                # print(f"{k}: test - {out_res[k]} вероятность: {prob_res[k]}")
            if isinstance(resHtml, str):
                print("Ready")
                return resHtml
                # return render_template("index.html", prediction_text=resHtml)

        else:
            raise Exception

    except TypeError as t:
        print("Попробуйте другой скан паспорта, похоже это не паспорт", t)
        return render_template(
            "index.html", prediction_text="Попробуйте другой скан паспорта"
        )

    except Exception as tt:
        print("Невыявлены значения строк, res - None,  попробуйте другой скан!!!")
        return render_template(
            "index.html",
            prediction_text="Невыявлены значения строк, res - None,  попробуйте другой скан!!!",
        )


if __name__ == "__main__":
    app.run(debug=True)
