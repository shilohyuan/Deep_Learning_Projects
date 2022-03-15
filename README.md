# Deep_Learning_Projects
Predicting COVID Patients from Chest X-Rays
Problem Statement  
This project aims to reliably identify COVID-19 through chest X-Ray images using transfer learning and even transformers to boost efficiency in COVID-19 management, especially before the potential outbreak of OMICRON variant  
Keywords: COVID-19 Management, Transfer Learning , VGG19 , Model FineTuning  
Data Preprocessing  
The dataset contains a total of 6939 image files divided equally into 3 classes (covid, pneumonia, normal)  
Resized into 128 x 128 pixels in order to reduce computational complexity  
Split the data into 80% train and 20% test sets   
Create a model with transfer learning  
Methodology  
Explan your Deep Learning process / methodology  

Transfer learning is: model developed for a task is reused as the starting point for a model on a second task  
Introduce the Deep Neural Networks you used in your project  

Model 1: Plain VGG 19 model adapted adding an output layer  
Model 2: VGG19 Finetuning adding additional conv2D layers  
Model 3: VG19 Extended Plus Data Augmentation  
Add keywords  
Keywords: multi-label classification  

MultiClass Evaluation against Extended VGG19 Model  
Recall is the porpotion of the positive is corectly classified which is true positive over all positive (true positive + false negative)  
Precision is the porpotion of predicted positives is truly positive which is true positive over the all predicted positive  
F1 is the weighted average of precision and recall, which is a combination metrics  

Issues / Improvements  
Dataset is very small (128 x 128 px)  
Need more hypertuning (epochs = 20)  
Lack of cross-validaiton  
References  
Chaddad, A., Hassan, L., & Desrosiers, C. (2021). Deep CNN models for predicting COVID-19 in CT and x- ray images. Journal of Medical Imaging, 8(S1), 014502. ​

https://www.kaggle.com/jefinpaul/covid-pneumonia-and-normal-using-resnet50-xray​

https://www.kaggle.com/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset​
