# path to weights models
path_first_model = r"C:\ml\Codes\RECOG_DOCS\test_from_server154_fordel\scanDoc\models\recog_first_pg_yolo_v8\weights\best.pt"
path_turn_model = r"C:\ml\Codes\RECOG_DOCS\test_from_server154_fordel\scanDoc\models\turn_first_pg_psp\weights\pretr_xception_09-val_acc-1.0000_val_loss-0.0004.hdf5"
# model recog rows passp
path_recog_row_model = r"C:\ml\Codes\RECOG_DOCS\test_from_server154_fordel\scanDoc\models\recog_row_model\best.pt"  # r"runs/detect/train/weights/best.pt"
# path model CRNN for recognize text from image
path_crnn_model = r"C:\ml\Codes\RECOG_DOCS\test_from_server154_fordel\scanDoc\models\model_crnn_txt\modelcrnn_906.keras"


height = 224
width = 224


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
#  C:\ml\Codes\RECOG_DOCS\test_from_server154_fordel\scanDoc\models
