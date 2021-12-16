## ROB535 Perception Image Classification Task Fall 2021
## Instructions
First get this shared folder
https://drive.google.com/drive/folders/1_mCLTbzKZMPgpDidFlrEOqSG8OuqFihU?usp=sharing

Then right click on the main "project" folder, click "add shortcut to drive"
![image](https://user-images.githubusercontent.com/39851166/146440402-cee729e2-19f8-443c-9856-3b557753e537.png)

If you directly add it to drive, you don't have to modify the colab file
Otherwise, you would need to change one paraemeter in section 0-3: change GOOGLE_DRIVE_PATH_AFTER_MYDRIVE and add the folder you added your file
### Introduction

- Tasked with correctly classifying vehicle and other modes of transport images generated from a Game engine. Developed an ensemble of soft voting Image classifiers and Object detectors to achieve a maximum test set accuracy of 0.78 on the Kaggle leaderboard. Our team ranked 2 in the respective ROB535 Kaggle competition 2021.

- The dataset contained 10204 snapshots out of which 7573 were used for training and the remaining 2631 for testing. From initial EDA we understood the heavy imbalance in the dataset, severe bias involved in images as many of them had very high auto correlations among each other. The training and test set distributions were very similar  and random samples indicated that Data augmentations would help us develop robust features and improve performance. 

### Methodology

1. *Stacked Soft Voting Image Classifier using EfficientNetV2s*

- The first step was to employ transfer learning using ResNetV250 and EfficientNetB4 models with no fine tuning on base layers to get the baseline performance. Leveraging the insights from EDA, programmed reference functions to create a directory structure with some classes clubbed together - effectively changing the output label from 23 dim to 8 dim. This helped us in effective error analysis and post processing was performed to output the required class labels.
  
- The images were augmented on the fly using Tensorflow ImageDataGenerator class. With our current dataset size, full fine-tuning of EfficientNetV2s model gave us the best accuracy of close to 0.78 on val set (15% split). The models were trained using Colab Pro and Google cloud platform. Identification of the limitations classes using confusion matrix was a key step to develop sub Image classifiers just on Cars, Utility and Off-road vehicles. 

- An EfficienNetV2s model was full fine tuned for over 75 epochs on the subclasses. Effective callbacks were used to save the model checkpoints and decay the learning rate. The stored models were loaded and a stacked soft voting classifier pipeline was designed. We effectively augmented each test set image with 5 random augmentations and took the average of the prediction probability to classify each test set image. We take the effective image class returned and programmed an if-else logic to split between full 8 class model or sub 3 class image classifier. In total 3 EfficientNetV2s image classifiers were employed to provide a final label and generate a csv file with an additional flag column. The flag column was binary with it being 1 when the image class was predicted as either Cars, Utility or Off-road class — this is where the object detection pipeline was performing better than the Image classifier and hence given priority in prediction.

2. *Object Detection using Faster R-CNN with ResNEt101+FPN and Temporal Conv*

- During training the stacked soft voting image classifier, we noticed that in most cases, the target vehicle only occupies a small portion of the image. It would be more efficient if we can detect the target area first and crop it out before classification. For this reason, we modified a faster R-CNN network with ResNet101+FPN as backbone to build an object detector and classifier for our desired classes. Compared to the previous method in the last section, this method first proposes regions that may contain the target and only investigate the proposed regions. This feature makes this network much faster to train. 

- In total, we trained the model on 80,000 epochs on the training set using varying learning rates. During the final few epochs, we fixed the hyper parameters and used the entire train-val set to train. During testing, we noticed that the test set are taken from a video sequence. To further explore this temporal information, we developed a simple temporal convolution on the output labels. For example, if we are detecting a plane in the frame before and frame after, but not the middle frame, it is very likely that we are making a mistake on the middle frame.

3. *Final Ensemble model*

During evaluation, we found out that our 2 methods are performing well at different classes (see appendix, fig xxx confusion matrix). While the object detection network is better at predicting cars, utility, or off-road classes, the direct classifier performs better at other classes in the dataset where limited training samples are given in the training set. For this reason, we went ahead and developed an ensemble of the two methods based on the predicted classes.

## Appendix
### Confusion matrix on the evaluation set for the object detection network
![image](https://user-images.githubusercontent.com/39851166/146443284-01e96537-204c-46be-a2a5-51c1f1ed1c82.png)

### Examples of correct object detection & classification under normal conditions 
![download (4)](https://user-images.githubusercontent.com/39851166/146442923-ab616ba1-4f37-4bc7-9de0-9250d5b6efd5.png)
![download](https://user-images.githubusercontent.com/39851166/146442982-56535b32-bae1-463b-8f80-33575a356a9a.png)
![download (3)](https://user-images.githubusercontent.com/39851166/146442990-a8fee858-3417-4416-ae45-0178803d81c9.png)


### Examples of correct object detection & classification under poor lighting, partial occlusions, or multiple targets
![download (5)](https://user-images.githubusercontent.com/39851166/146442908-e732feb0-30bc-4c48-8c8c-227d473f45f4.png)
![download (2)](https://user-images.githubusercontent.com/39851166/146442952-8117ffd7-9f64-40e2-8f5e-2f461ea13910.png)
![download (1)](https://user-images.githubusercontent.com/39851166/146442972-035be039-2ced-49fd-bbcf-d93634ec022a.png)

### Uneven distribution between classes
![image](https://user-images.githubusercontent.com/39851166/146443228-0527adc6-bcc5-4bff-85ef-a6359727baf7.png)


