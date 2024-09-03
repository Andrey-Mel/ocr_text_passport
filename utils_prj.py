import numpy as np
import pandas as pd
import os
import cv2
import keras
from pathlib import Path
from shutil import copyfile
import tensorflow as tf
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

from utils_func import get_recog_txt  # , crft_init

from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    BatchNormalization,
    GlobalAveragePooling2D,
    Activation,
    Input,
    Flatten,
)
from tensorflow.keras.layers import MaxPooling2D, Dropout

from skimage.transform import resize
from skimage.color import rgb2gray

import ultralytics
from ultralytics import YOLO

ultralytics.checks

import matplotlib.pyplot as plt


def create_model(freez_batchNorm_only=False):

    base_model = tf.keras.applications.xception.Xception(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    global_average = GlobalAveragePooling2D(name="Clobal_layer")(x)  # None, 1024
    drop = Dropout(0.3, name="Dropout_layer")(global_average)
    dense = Dense(1024, activation="relu", kernel_initializer="glorot_uniform")(drop)
    batchNorm = BatchNormalization()(dense)
    out = Dense(5, activation="softmax")(batchNorm)

    model = Model(inputs=base_model.input, outputs=out)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["categorical_accuracy"],
    )

    if freez_batchNorm_only == False:
        base_model.trainable = False  # замораживаю предобученную модель полностью
    elif freez_batchNorm_only == True:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalizatioin):
                layer.trainable = False
            else:
                layer.trainable = True

    return model


def preprocess_img(path_img, shape=(224, 224)):
    img = image.load_img(path_img, target_size=shape)
    img = image.img_to_array(img)
    img = np.array(img) / 255
    img = np.expand_dims(img, axis=0)
    return img


# функция получения класса с предикта
def get_class(model, img_arr):
    print("Get angle...")
    class_map = ["0_norm", "1_left", "2_right", "3_down", "4_not_pg"]
    out = class_map[np.argmax(model(img_arr))]
    return out


# функция на полученных данных поворот фото
def correct_photo(img_arr, class_turn):
    """
    Поворот фото по массиву
    Возвращает тип PIL.Image.Image
    Если нужно в массив то image.img_to_array
    """
    print("Correct image...")
    img_arr = (img_arr.squeeze()).astype(np.uint8)
    # print(' - ', img_arr.shape, class_turn)
    image = Image.fromarray(img_arr)
    if class_turn == "1_left":
        image = image.rotate(-90, expand=True)
    elif class_turn == "2_right":
        image = image.rotate(+90, expand=True)
    elif class_turn == "3_down":
        image = image.rotate(180, expand=True)
    elif class_turn == "4_not_pg":
        print("None page")
        return None
    else:
        image = image
    return image


def load_model(path_weight):
    print("Create turn model...")
    model = create_model()
    try:
        model.load_weights(path_weight)
    except:
        print("Not load scanDoc model!")
    # assert isinstance(model, keras.engine.functional.Functional), "Not load model("
    print("Ok!")
    return model


def preproc_im(crop, shape=(224, 224)):
    print("Prepair image in model")
    crop_img_rz = tf.image.resize(crop, shape) / 255
    return np.expand_dims(crop_img_rz, axis=0)


def correct_page(path, recog_model, model_xcept, height=224, width=224):
    """
    recog_model: модель распознавания 1 страницы паспорта
    model_except: модель поворота первой страницы паспорта
    """
    path = os.path.normpath(path)
    assert os.path.isfile(path), "File is not find"
    print("CORRECT PAGE PATH - ", path)
    img = Image.open(path)
    img.show()
    result = recog_model(path, conf=0.9, verbose=False)
    img_arr = result[0].orig_img
    boxes = result[0].boxes.cpu().numpy()
    crop_img = np.array([], dtype=np.int8)
    if len(boxes) != 0:
        for box in boxes:

            coord = box.xyxy[0].astype(int)
            crop_img = img_arr[coord[1] : coord[3], coord[0] : coord[2]]
    # if there is one page on the scan, then we take the main one scan img
    else:
        crop_img = img_arr

    print("CROPED image shape {0}".format(crop_img.shape))
    plt.imshow(crop_img, cmap="gray")
    plt.show()

    # Для контроля
    # crop_img_vis = Image.fromarray(crop_img)

    img_to_model = preproc_im(crop_img, shape=(width, height))
    cl_turn = get_class(model_xcept, img_to_model)
    if cl_turn == "4_not_pg":
        print("Cl_turn - 4, none page")
        return None
    else:
        image = np.asarray(correct_photo(crop_img, cl_turn), dtype=np.uint8)
        return image


# YOLO first model
def get_recog_model(path):
    print("Create recog first page model...")
    if os.path.isfile(path):
        model = YOLO(path)
        assert isinstance(model, ultralytics.models.yolo.model.YOLO), "Not create model"
        print("Ok!")
    else:
        raise Exception("Not find file with weigth model")
    return model


def prep_img(img):
    img_height = 50
    img_width = 200

    if img.shape[0] < img.shape[1]:
        # img = np.rot90(img, 1)
        img = np.transpose(img, (1, 0, 2))
    else:
        img = np.fliplr(img)
    img = rgb2gray(img)
    img = resize(img, (img_width, img_height, 1), anti_aliasing=True)
    return img


def groping(coordinats):
    (
        lst_10,
        lst_20,
        lst_30,
        # lst_40,
        # lst_50,
        # lst_60,
        lst_70,
    ) = ([], [], [], [])
    all_data = []
    out_data = []
    for i in coordinats:
        if i[2] <= 20:
            lst_10.append(i)

        elif 20 < i[2] <= 40:
            lst_20.append(i)

        elif 40 < i[2] <= 60:
            lst_30.append(i)

        # elif 30 < i[2] <= 40:
        #     lst_40.append(i)

        # elif 40 < i[2] <= 50:
        #     lst_50.append(i)

        # elif 50 < i[2] <= 60:
        #     lst_60.append(i)
        else:
            lst_70.append(i)

    all_data = [
        lst
        for lst in (
            lst_10,
            lst_20,
            lst_30,
            # lst_40,
            # lst_50,
            # lst_60,
            lst_70,
        )
        if lst
    ]
    for i in range(len(all_data)):
        group_sort = sorted(all_data[i], key=lambda point: (point[0]))
        out_data.extend(group_sort)
        # out_data.extend(all_data[i])
    return out_data


# model_crft = crft_init()


def row_fpp(image: np.ndarray, model, model_crft, confid=0.5) -> dict:
    """Recognition row first page passp
    image: np.array image fpp
    confid: confidence
    model: model rocognition fields on pasp first page
    model_crft: model craft recognize words on fields pasp first page
    """

    row_data = {
        "паспорт выдан": [],
        "серия, номер": [],
        "дата выдачи": [],
        "код подразделения": [],
        "фамилия": [],
        "имя": [],
        "отчество": [],
        "пол": [],
        "дата рождения": [],
        "место рождения": [],
    }
    class_m = {
        0: "паспорт выдан",
        1: "серия, номер",
        2: "дата выдачи",
        3: "код подразделения",
        4: "фамилия",
        5: "имя",
        6: "отчество",
        7: "пол",
        8: "дата рождения",
        9: "место рождения",
    }

    res = model.predict(image, conf=confid, verbose=False)
    image = res[0].orig_img
    for i in range(len(res)):
        arim = pd.DataFrame(res[i].boxes.data).astype(float)
    arim.columns = ["x", "y", "x2", "y2", "confidence", "class"]
    arim["class"] = arim["class"].apply(lambda x: class_m[int(x)])
    for i in range(len(arim)):
        label = arim.iloc[i, 5]

        conf = str(round(arim.iloc[i, 4], 2))
        # print(f" {label} = {conf}")
        x = int(arim.iloc[i, 0])
        y = int(arim.iloc[i, 1])
        x2 = int(arim.iloc[i, 2])
        y2 = int(arim.iloc[i, 3])

        if label == "паспорт выдан":
            img_psp_lst = []
            img_psp = np.array(image[y:y2, x:x2])
            _, coord = get_recog_txt(model_crft, img_psp)  # img_psp_lst
            # for sorted recognize images from "паспорт выдан"
            coord_sorted = sorted(
                coord, key=lambda point: (point[2], point[0], point[1], point[3])
            )
            out_coord = groping(coord_sorted)
            for i in range(len(coord_sorted)):
                x = out_coord[i][0]
                x2 = out_coord[i][1]
                y = out_coord[i][2]
                y2 = out_coord[i][3]
                img_psp_lst.append(img_psp[y:y2, x:x2])
            # prepering recog sorted imgs
            imgs_psp_txt = [prep_img(np.array(img)) for img in img_psp_lst]
            row_data["паспорт выдан"].extend(imgs_psp_txt)

        elif label == "серия, номер":
            img_sn = prep_img(np.array(image[y:y2, x:x2]))
            row_data["серия, номер"].append(img_sn)

        elif label == "дата выдачи":
            img_dv = prep_img(np.array(image[y:y2, x:x2]))
            row_data["дата выдачи"].append(img_dv)

        elif label == "код подразделения":
            img_cp = prep_img(np.array(image[y:y2, x:x2]))
            row_data["код подразделения"].append(img_cp)

        elif label == "фамилия":
            img_f = prep_img(np.array(image[y:y2, x:x2]))
            row_data["фамилия"].append(img_f)

        elif label == "имя":
            img_n = prep_img(np.array(image[y:y2, x:x2]))
            row_data["имя"].append(img_n)

        elif label == "отчество":
            img_o = prep_img(np.array(image[y:y2, x:x2]))
            row_data["отчество"].append(img_o)

        elif label == "пол":
            img_s = prep_img(np.array(image[y:y2, x:x2]))
            row_data["пол"].append(img_s)

        elif label == "дата рождения":
            img_dr = prep_img(np.array(image[y:y2, x:x2]))
            row_data["дата рождения"].append(img_dr)

        elif label == "место рождения":
            img_mr_lst = []
            img_mr = np.array(image[y:y2, x:x2])
            # plt.imshow(img_mr)
            # plt.show()
            _, coord_mr = get_recog_txt(model_crft, img_mr)
            # for sorted recognize images from "паспорт выдан"
            coord_sorted_mr = sorted(
                coord_mr, key=lambda point: (point[2], point[0], point[1], point[3])
            )
            out_coord_mr = groping(coord_sorted_mr)
            for i in range(len(out_coord_mr)):
                x = out_coord_mr[i][0]
                x2 = out_coord_mr[i][1]
                y = out_coord_mr[i][2]
                y2 = out_coord_mr[i][3]
                # plt.imshow(img_mr[y:y2, x:x2])
                # plt.show()
                img_mr_lst.append(img_mr[y:y2, x:x2])

            img_mr_txt = [prep_img(img) for img in img_mr_lst]
            row_data["место рождения"].extend(img_mr_txt)

    return row_data


# ctc_decode записей
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = tf.shape(y_pred)  # batch, 50, 52(vocab)
    num_samples, num_steps = input_shape[0], input_shape[1]  # batch, 50
    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon()
    )  # perm = 50, batch, 52
    input_length = tf.cast(input_length, tf.int32)  # array batch x 50

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
            merge_repeated=True,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))

    return decoded_dense, log_prob


# Decode


characters = [
    " ",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "T",
    "Ё",
    "А",
    "Б",
    "В",
    "Г",
    "Д",
    "Е",
    "Ж",
    "З",
    "И",
    "Й",
    "К",
    "Л",
    "М",
    "Н",
    "О",
    "П",
    "Р",
    "С",
    "Т",
    "У",
    "Ф",
    "Х",
    "Ц",
    "Ч",
    "Ш",
    "Щ",
    "Ъ",
    "Ы",
    "Ь",
    "Э",
    "Ю",
    "Я",
    "№",
]

# characters to integer
char_to_num = tf.keras.layers.StringLookup(
    vocabulary=list(characters),
    mask_token=None,
    max_tokens=None,
    pad_to_max_tokens=None,
)

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def invert2text_out(num):
    out_idx = num[num != -1]
    res = tf.strings.reduce_join(num_to_char(out_idx)).numpy().decode("utf-8")
    return res


def decode_batch_prediction(pred, greedy=True):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]  # aray bacth x 50
    results, log_prob = ctc_decode(
        pred, input_length=input_len, greedy=greedy
    )  # [:,:max_length]
    output_text = []
    return results, log_prob  # output_text


def output_res(res, recog_txt_model):
    out = {}
    probability = {}
    print("FUNCTION OUTPUT RES...")

    for k, v in res.items():
        if k == "паспорт выдан":
            sentence = ""
            probs_lst = []
            for d in res[k]:
                pred = recog_txt_model.predict(d[None,], verbose=0)
                dec_it, prob_log = decode_batch_prediction(pred)
                # collect sentence string in dict out
                sentence += str(invert2text_out(dec_it[0])) + " "
                # calc probability
                probs_lst.append(np.round(prob_log[0].numpy()[0], 2).astype(np.float16))
                max_probs = np.max(probs_lst)
                # to visual
                out[k] = sentence
                probability[k] = max_probs  # mean_probs

        elif k == "место рождения":
            sentence = ""
            probs_lst_mr = []
            for m in res[k]:
                pred = recog_txt_model.predict(m[None], verbose=0)
                dec_it, prob_log = decode_batch_prediction(pred)
                sentence += str(invert2text_out(dec_it[0])) + " "
                # probs_lst_mr.append(np.round(tf.exp(prob_log[0]).numpy()[0], 2).astype(np.float16))
                probs_lst_mr.append(
                    np.round(prob_log[0].numpy()[0], 2).astype(np.float16)
                )
                max_probs_mr = np.max(probs_lst_mr)
                out[k] = sentence
                probability[k] = max_probs_mr  # mean_probs
        else:

            pred = recog_txt_model.predict(res[k][0][None,], verbose=0)
            dec_it, prob_log = decode_batch_prediction(pred)
            out[k] = invert2text_out(dec_it[0])
            # write in dict probability
            probability[k] = np.round(prob_log[0].numpy()[0], 2).astype(np.float16)
    return out, probability
