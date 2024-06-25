# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developed by Thilo Schwind for Udacity ML DevOps Training on 18.06.2024, Version 1
- DecisionTreeClassifier by scikit-learn 

## Intended Use
- Intended to be used to apply the skills acquired in course "Deploying a Scalable ML Pipeline in Production"
## Training Data
- Kohavi,Ron. (1996). Census Income. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.
- training data split

## Evaluation Data
- Kohavi,Ron. (1996). Census Income. UCI Machine Learning Repository. https://doi.org/10.24432/C5GP7S.
- test data split

## Metrics
- Prediction were made with following parameters:

    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       random_state=42, splitter='best')

- Evaluation metrics include **precision**, **recall** and **fbeta**.
- **Precision**: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0. (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- **Recall**: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0. (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#recall-score)
- **fbeta**: The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0. (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#fbeta-score)
- For evaluation on test data set followinf metric scores wre obtained: 
    - precision: 0.617, 
    - recall: 0.620, 
    - fbeta: 0.618
## Ethical Considerations
- Predictions could lead to false assumptions about the income of minorities, resulting in false prejudices or the like. 
## Caveats and Recommendations
- Model shows different performance metrics depending on slices, such as "native-country" (eg. Model performance for Mexico: precision: 0.125, recall: 0.143, fbeta: 0.13) or education (eg. Model performance for 7th-8th: precision: 0.111, recall: 0.2, fbeta: 0.143)
