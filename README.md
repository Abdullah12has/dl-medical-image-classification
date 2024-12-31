
Transfer Learning for Medical Image Classification
     Instructed  by Haotian Liu (haotian.liu@oulu.fi)

Motivation: In medical imaging, obtaining large, labeled datasets is often challenging due to privacy concerns, high annotation costs, and limited availability of expert knowledge. To effectively learn and boost performance on these smaller datasets, we leverage transfer learning techniques, which consist of models that are trained on huge amounts of data.
Goal: Improve the performance of diabetic retinopathy detection using transfer learning by fine-tuning models and understanding the classification results with visualizations and explainable AI.
Requirements:

     1. Complete the project and submit the code. For the code, you can get help from github. (45 points)

          The DeepDRiD dataset, template code, and online evaluation are available on Kaggle: https://www.kaggle.com/t/41e0944a6839469fadd529fabab45e06. You should only use this dataset for final project purposes.

          You are required to complete a few tasks as follows. Please check the instruction document below for more information.

          a) Fine-tune a pre-trained model using the DeepDRiD dataset. (5 points)

          b) Two-stage training with additional datasets. (5 points)

          c) Incorporate attention mechanisms in the model. (10 points)

          d) Compare the performance of different models and strategies. (20 points)

          e) Creating Visualizations and Explainable AI. (5 points)

     2. Submit a report for this project. (15 points)

          The report should include a description of the methods used in the project, experimental results, and discussions.


          Please include your team name(s) of Kaggle in the Google form above. The report should include the results of the DeepDRiD test set.


          The plagiarism check rate of the report needs to be less than 20% (excluding references). 



Hints:
For GPU computation resources, you can choose to use the CSC student project, Google Colab, and computer rooms in TS135.
We encourage students to get code inspiration from GitHub. Please submit the zip file, which includes your code, trained model, and a readme.md file (explaining how to run the code). We also accept the code in notebook or Python script form.
Students can create a Google link, upload the trained checkpoints there, and send the link when they submit the final project. You can choose to put the link in the report.


___________________________________________________________________

Things we tried


Part a
1. First tried to run all models including resnet18, resnet34, resnet 50, vgg, densenet in single and dual mode. we got the best output from vgg in single mode that was 83% kappa. 
2. We tried to play with different preprocessing techniques like rotating the images, changing the color, applying gaussian noise, and other things like flips, Elastic TransformationsAutoAugment: Use libraries like torchvision.transforms.AutoAugment or RandAugment for automatic augmentation policies.
* CutMix/MixUp: These techniques blend images and labels, improving generalization. But mostly things resulted in worse results rather than better results. 


We started to systematically test the learning rate of models, we A/B tested reducing and increasing. Then we tested dropout rate with different values like 52, 49, 48, 60. We got better results at 52 with vgg, and resnet 50: 0.8419 kappa. 

Then we tried running all models with dual mode to see difference in models with dual mode. We got: ResNet18 Base single mode: Best kappa: 0.8066, Epoch 19
Dual Mode: Best kappa: 0.7777, Epoch 17

 resnet34: Best kappa: 0.7758, Epoch 9
 resnet34 Dual Mode Best kappa: 0.8088 Epoch 5


Then we tried to visualize the data to understand why are we getting relatively low results for class 3,4. We realized after visualizing the data that the samples of class 0 are the most in the dataset and other classes are relatively under represented. So we tried to run SMOTE but that’s not possible with image data so we though maybe it’ll be a good idea to Dynamic Oversampling: Use weighted sampling to address imbalance during training.
We had our base results for res18: Best kappa: 0.7777,So we ran res18 with the dynamic oversampling for 10 epos: we got Best kappa: 0.7618, Epoch 7. So dynamic oversampling was not the answer. 



For part B, we unfroze all the layers of the model and loaded the weights of vgg and used the vgg model. We got results of around 86.35. We also removed fundrandomrotate from the preprocessing. 


We tried to keep all the layers unfrozen except the last 5 layers: we got: 83.56 

Then we tried to only unfreeze last 5 layers, we got: 82.81% not very good. 


Part c: we implemented spatial attention. 	
•	because these mechanisms help the model focus on important regions of the retina, such as lesions or abnormalities.
Result: 83.17% not a very impressive improvement


tried self attention mechanism improved a little to : 84. 42%





resnet 18: tried unfrozen layers and self attention mechanism with preloaded weights on the big dataset and then fine tuned on the main dataset: we got 83.98 and 83.49


part d... Stacking, Boosting, Weighted Average, Max Voting, Bagging from the results of the 3 models



Evaluating Weighted Average Ensemble...
Weighted average prediction: 100%|█████████████████████████████████████████████████| 17/17 [00:03<00:00,  4.70it/s] 

Weighted Average Metrics:
Accuracy: 0.6800
Kappa: 0.8043
Precision: 0.6626
Recall: 0.6800

Evaluating Max Voting Ensemble...
Max voting prediction: 100%|████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  8.92it/s] 

Max Voting Metrics:
Accuracy: 0.6600
Kappa: 0.7742
Precision: 0.6471
Recall: 0.6600

Evaluating Stacking Ensemble...
Training stacking ensemble...
Extracting features: 100%|██████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.90it/s] 
Extracting features: 100%|██████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  8.72it/s] 

Stacking Metrics:
Accuracy: 0.6400
Kappa: 0.7941
Precision: 0.6367
Recall: 0.6400

Evaluating Boosting Ensemble...
Training boosting ensemble...
Extracting features: 100%|██████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.62it/s] 
Extracting features: 100%|██████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  8.85it/s] 

Boosting Metrics:
Accuracy: 0.6650
Kappa: 0.8063
Precision: 0.6646
Recall: 0.6650

Evaluating Bagging Ensemble...
Training bagging ensemble...
Extracting features: 100%|██████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  7.68it/s] 
Extracting features: 100%|██████████████████████████████████████████████████████████| 17/17 [00:01<00:00,  8.77it/s] 

Bagging Metrics:
Accuracy: 0.6525
Kappa: 0.8047
Precision: 0.6585
Recall: 0.6525



 ## On test cases: 

Bagging: 79.60
Boosting: 78.61
Max_voting: 85.28
Stacking: 79.52
Weighted Predictions: 84.66 

The base models used for this had the following results on the test case:


Resnet18: 83.91%
Resnet34: 78.73%
VGG16: 84.17%



Part d-2: 
Image augmentation techniques

Base model: VGG16 with spacial attention and unfrozen layers

BASE MODEL Performance: 83.58.

Will try:
Ben  Graham,  Circle  Cropping, CLAHE, adding gaussian blur, sharpening up the images to see difference on results

ben Graham: 28.49 really bad
SharpeningTransform():79.93
gaussian blur: 80.10
CLAHETransform:83.82
Circle Crop Transform: 82.59
All: 84.29

applying all improved...

removing sharpening, and gaussian blur and trying again: 81.70












______________________________________________________________________
WE TRIED TO DO ENSEMBLE WITH MULTIPLE INSTANCES OF THE SAME MODEL. Just to test would it increase the result.

# BAGGING

## resnet 18
implemented bagging ensemble with pretrained weights, and unfrozen layers + self attention mechanism. 5 model bagging: result:5 models with 2 ephos: 86.88 
tried same thing with 15 ephos: got 82.11

res18 tried 3 ephos with 10 models: 85.05
resnet 18 3 ephos with 3 models: 84.24

same: but without attention and with all layers frozen: 83.09%


means we should keep these 2 on. 


Now trying resnet 18 with dual mode with attention mechanism and unfrozen layers: bagging 5 models with 6 ephos per modelResult: 78.48. Not gonna try dual again. not worth it. 

## resnet 34
Next: tried resnet 34 with self attention mechanism and unfrozen layers: got 79.05 

tried to improve by adding focul loss function to criterion: res: 84.14 
## vgg 16
Tried VGG16 with bagging (all layers unfrozen, self attention added):res: 84:63


# BOOSTING

## VGG16
We implemented ensemble learning Boosting: we got 85.02% on the test case. We tried same boosting but on vgg that had frozen layers: we got 83.72% results 

We improved the code on vgg also with frozen layers again: we got: 80ish

We added an augmentation to vgg all layers unfrozen: 
transforms.RandomApply([transforms.Lambda(lambda img: adjust_gamma(img, gamma=1.5))], p=0.3),  # Gamma correction

- gamma correction and we got: 83.87

## Resnet 18

NOTE: ADDING ATTENTION TO LAYERS 3 and 4 instead of all layers actually increases the results

Resnet 18 with 20 ephos, 0.52 dropoff rate and ofc boosting
Result: 79.59%

Now trying another way of boosting:
integrates boosting by extracting features from a deep learning model and then applying a traditional gradient boosting algorithm (GradientBoostingClassifier from scikit-learn). 
We got: 82.13% Better than the previous one. 



## Resnet 34

Since the hybrid approach is working better, we will only apply that to resnet 34
Increased the batch size to 34: because it was giving better results


# Stacking


## VGG16
Training Best Kappa: 85,98
Test Results: 85,21

## Resnet 18
Training Best Kappa: 0.8680
Test Results: 0.8567

## Resnet 34
Training Best Kappa: 0.8472
Test Results: 0.8282

# Weighted Average

## VGG16

1. CONFIG={'models':{'vgg16':True,'resnet18':False,'resnet34':False},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':True,'max_voting':False,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa: 86.19
3. Kaggle Score: 79.55

## Resnet 18
1. CONFIG={'models':{'vgg16':False,'resnet18':True,'resnet34':False},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':True,'max_voting':False,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa: 96,38
3. Kaggle Score: 78,12

## Resnet 34
1. CONFIG={'models':{'vgg16':False,'resnet18':False,'resnet34':True},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':True,'max_voting':False,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa:87,82
3. Kaggle Score: 78,12

# Max Voting

## VGG16
1. CONFIG={'models':{'vgg16':True,'resnet18':False,'resnet34':False},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':False,'max_voting':True,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa: 
3. Kaggle Score: 76,28

## Resnet 18
1. CONFIG={'models':{'vgg16':False,'resnet18':True,'resnet34':False},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':False,'max_voting':True,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa: 
3. Kaggle Score: 71,26

## Resnet 34

1. CONFIG={'models':{'vgg16':False,'resnet18':False,'resnet34':True},'ensemble_methods':{'stacking':False,'boosting':False,'weighted_average':False,'max_voting':True,'bagging':False},'preprocessing':{'ben_graham':True,'circle_crop':True,'clahe':True,'gaussian_blur':True,'sharpen':True}}
2. Training Best Kappa: 
3. Kaggle Score: 78,84



