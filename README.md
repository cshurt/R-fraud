# R-fraud
Class project in which I utilized different classifying algorithms on a Kaggle credit-card-fraud dataset.

## Results
Logistic regression was the most impressive classifier by the AUC measure, with Random Forest tree aggregation and a shallow Neural Network also having very similar results.

## The data
284,807 transactions, of which 492 were known as 'ground truth' fraudulent. Each transaction is described by 28 features (including time and amount). The data has been anonymized via application of principal component analysis (among other things) with exception to the amount spent in each transaction.

## Methodology
Using R, I divided the data into training, verification, and testing sets. I split both the fraudulent and non-fraudulent transactions, to ensure 'ground truth' fraud was in every group. With such a sparse classifier, I used the AUC as the principal measurement. I trained and validated four models, a single decision tree, logistic regression, single-hidden-layer neural network, and a Random Forest tree combination. Each was trained to maximize the AUC, then verified to tune hyperparameters, and then tested on the test set.

## Future Research
If I were to continue on this project, I would compare the AUC to the precision-recall curce. Due to the importance of classifying True Positives in the very sparse fraud data, the precision-recall curve would be uniquely suited as a target. Additionally, it would be interesting to allow machine learning techniques more say in the creation of features. I would be very interested in applying a recurrent neural network as a form of deep learning selection of classifiers. This would be impossible on this dataset (as the data has been anonymized) but might be of interest to companies with real data.

## Contents
All code is contained in the Fraud\_Detection\_Measurement.R file. The SHURTLEFF-POSTER file contains a high-level summary of some of the results. The dataset can be downloaded [here](https://www.kaggle.com/dalpozz/creditcardfraud).
