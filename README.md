# R-fraud
Class project in which I utilized different classifying algorithms on a Kaggle credit-card-fraud dataset.

## Results
Logistic regression was the most impressive classifier by the AUC measure, with Random Forest tree aggregation and a shallow Neural Network also having very similar results.

## The data
284,807 transactions, of which 492 were known as 'ground truth' fraudulent. Each transaction is described by 28 features (including time and amount). The data has been anonymized via application of principal component analysis (among other things) with exception to the amount spent in each transaction.

## Methodology
Using R, I divided the data into training, verification, and testing sets. I split both the fraudulent and non-fraudulent transactions, to allow each
