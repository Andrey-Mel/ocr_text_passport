# OCR TEXT PASSPORT

INTRODUTION.

The project is designed to recognize the data of the first page of the Russian passport from a scanned passport or photo. Since the project is related to the passport, I cannot demonstrate examples of the result, for well-known reasons. But you can try it yourself. The project demonstrates how you can apply pre-trained models, as well as write your own and use them for the purpose you need. 

DESCRIPTION OF THE PROJECT.

The project is not a research work, but rather an example of the application of an ensemble of models. The YOLOv8, Xception, CRAFT models are involved, and its architecture is written in the keras framework.
	YOLOv8 is the repository of this model ultralytics.com . It is used twice, but it is trained to detect each its own purpose. The first step is the detection of the first page of the passport, if the passport scan consists of several pages. The second task of detection is to detect the fields of the passport (by whom it was issued, when it was issued, full name, serial number, unit code, place of birth, date of birth).
	Xception with a modified model head, used for the correct vertical reversal of the first page of the passport. Taking into account the fact that a certain first page of the passport (the first step) can be located differently on the plane. Pre-trained weights were taken and therefore the accuracy of the test after pre-training reached 98% accuracy.
The next step is for the YOLOv8 model to determine the passport fields we need and we get 10 small images with text in the fields.
	A CRAFT model for defining words in images where the text reaches 3 lines, these are fields such as "Issued by whom" and "Place of birth". This model is based on the definition of letters using a Gaussian intensity map (GAUSSIAN HEATMAP), here is the link - https://habr.com/ru/companies/inDrive/articles/598193 / to the article where the operation of this model is described in great detail. https://github.com/clovaai/CRAFT-pytorch/tree/master - this is the repository of the model.
After the release of the CRAFT model, we have separate images of words for each required field of the passport, since I took CTCLoss of the keras framework to recognize text from an image, and I wrote a model for recognizing text from images on keras. I translated images with words to the dimension 200,50,1. The model consists of convolutional and recurrent layers. In testing after training, this model showed a result in 90% of correctly recognized images with words and numbers. 
	The result of the model is decoding the output from CTCLoss, and we get word recognition, then we collect these words into the string variable, also the value of the recognized word comes out of CTCLoss, the closer to zero the more accurately the recognized text.
	This project does not use a transformer, according to the articles it has better recognition of text from an image than the CTCLoss model.
 
USE.

If you look at the results in the browser, you need to run the python file app.py . But since I am not a frontend developer, there is no beautiful interface in the browser, there is only a photo search button and a predict button.
