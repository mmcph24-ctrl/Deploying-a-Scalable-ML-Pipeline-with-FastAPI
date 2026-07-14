# Model Card

## Model Details

This project uses a Random Forest classification model built with scikit-learn. The model predicts whether a person's salary is `>50K` or `<=50K`. Categorical features are processed with one-hot encoding before training.

## Intended Use

The model was created for an educational machine learning project. It demonstrates data preprocessing, model training, evaluation, slice analysis, and deployment with FastAPI.

Disclaimer: The model should not be used for high-stakes decisions involving employment, lending, insurance, or eligibility.

## Training Data

The model was trained using the Census Income dataset provided with the project. The dataset contains 32,561 records, with `salary` used as the target variable.

The data was split into 80 percent for training and 20 percent for testing using a random state of 42. Categorical features were processed using one-hot encoding.

## Evaluation Data

The test dataset contains 6,513 records and was processed using the encoder created from the training data.

The model was also evaluated on each unique value within the categorical features. These results are stored in `slice_output.txt`.

## Metrics

The model was evaluated using precision, recall, and F1 score.

Overall model performance:

* Precision: 0.7419
* Recall: 0.6384
* F1 score: 0.6863

The slice results show that model performance varies across different categorical groups.

## Ethical Considerations

The dataset includes demographic features such as race and sex. Because the data may reflect existing social and economic inequalities, the model could also learn those patterns.

Disclaimer: The model is intended for educational purposes only and should not be used to make decisions that directly affect individuals.

## Caveats and Recommendations

The salary classes are imbalanced, with more `<=50K` records than `>50K` records. Some categorical values also have small sample sizes, which can affect slice metrics.

Model performance should be reviewed using both the overall metrics and the results in `slice_output.txt`.
