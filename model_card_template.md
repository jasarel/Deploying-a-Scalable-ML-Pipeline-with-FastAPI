# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model uses a Random Forest Classifier using scikit-learn on the Census income data set. The model was developed by Jasmine J Arellano.
## Intended Use
The model is intended to determine whether an individual makes above or below $50,000 per year. This model follows a full ML pipeline, including training, testing, evaluation, deployment with FastAPI, CI/CD integration using DVC and GitHub Actions.
## Training Data
The training data used was from the Census Income Data Set and included columns such as age, marital status, education, occupation, salary, etc.
## Evaluation Data
The training data is divided into slices, using approximately 20% of the data.
## Metrics
The metrics used to test the models performance were: 
Precision, which was scored at 0.73,
Recall scored at 0.59,
and an F1 score of 0.65.

## Ethical Considerations
Since the model is trained on real-life data, there could be some bias in the data set such as there being social bias if there are more people from one country than the other. 

Another consideration to think about is how some of the information is sensitive, such as education, salary, race, etc. 
## Caveats and Recommendations
Some things to look out for include ensuring the data is up-to-date (Income is reflected accurately) and that bias mitigation strategies are in place which can include retraining or extra validation. 
