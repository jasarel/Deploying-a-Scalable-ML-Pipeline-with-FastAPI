import pytest
# TODO: add necessary import
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split

# TODO: implement the first test. Change the function name and input as needed
def test_model_algorithm():
    """
    # This test will ensure that the model uses the expected algorithm
    """
    # Create sample data 
    X = [[0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1]

    model = train_model(X,y)

    assert isinstance(model, RandomForestClassifier), \
    f"Expected model to be RandomForestClassifier, got {type(model).__name__} instead"


# TODO: implement the second test. Change the function name and input as needed
def test_metrics():
    """
    # This test will ensure that the computing metrics functions return the expected result  
    """
    # Create sample data
    y = [0, 1, 1, 0]
    preds = =[0, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    """
    # This test ensures that the test and training sets are the correct size 

    """
    # Sample data
    X = [[0, 1], [1, 0], [1, 1], [0,0]]
    y = [0, 1, 1, 0]

    # Train model 
    model = train_model(X, y)

    # Run inference
    y_preds = inference(model, X)

    assert y.shape == y_preds.shape, f"Expected shape to be {y.shape}, but got {y_preds.shape}"

