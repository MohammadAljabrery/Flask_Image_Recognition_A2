import os
import pytest
import numpy as np
from keras.models import load_model
from model import preprocess_img, predict_result  # Adjust based on your structure

# Load the model before tests run
@pytest.fixture(scope="module")
def model():
    """Load the model once for all tests."""
    model = load_model("digit_model.h5")  # Adjust path as needed
    return model

# Basic Tests

def test_preprocess_img():
    """Test the preprocess_img function."""
    img_path = "test_images/2/Sign 2 (97).jpeg"
    processed_img = preprocess_img(img_path)

    # Check that the output shape is as expected
    assert processed_img.shape == (1, 224, 224, 3), "Processed image shape should be (1, 224, 224, 3)"

    # Check that values are normalized (between 0 and 1)
    assert np.min(processed_img) >= 0 and np.max(processed_img) <= 1, "Image pixel values should be normalized between 0 and 1"


def test_predict_result(model):
    """Test the predict_result function."""
    img_path = "test_images/4/Sign 4 (92).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)

    # Print the prediction for debugging
    print(f"Prediction: {prediction} (Type: {type(prediction)})")

    # Check that the prediction is an integer (convert if necessary)
    assert isinstance(prediction, (int, np.integer)), "Prediction should be equal to 5"

def test_predict_result_non_trained_image():
    """Test the model to see if it can accuretly predict images it was not trained on"""
    img_path = "test_images/validation_testing_images/5_test.jpg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 5
    assert 5 == prediction, "Prediction should be equal to 5"

def test_predict_result_0():
    """Test the model to see if it can accuretly predict an image of hand holding up 0"""
    img_path = "test_images/0/Sign 0 (116).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 0
    assert 0 == prediction, "Prediction should be equal to 0"

def test_predict_result_1():
    """Test the model to see if it can accuretly predict an image of hand holding up 1"""
    img_path = "test_images/1/Sign 1 (150).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 1
    assert 1 == prediction, "Prediction should be equal to 1"

def test_predict_result_2():
    """Test the model to see if it can accuretly predict an image of hand holding up 2"""
    img_path = "test_images/2/Sign 2 (87).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 2
    assert 2 == prediction, "Prediction should be equal to 2"

def test_predict_result_3():
    """Test the model to see if it can accuretly predict an image of hand holding up 3"""
    img_path = "test_images/3/Sign 3 (83).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 3
    assert 3 == prediction, "Prediction should be equal to 3"

def test_predict_result_4():
    """Test the model to see if it can accuretly predict an image of hand holding up 4"""
    img_path = "test_images/4/Sign 4 (92).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 4
    assert 4 == prediction, "Prediction should be equal to 4"

def test_predict_result_5():
    """Test the model to see if it can accuretly predict an image of hand holding up 5"""
    img_path = "test_images/5/Sign 5 (147).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 5
    assert 5 == prediction, "Prediction should be equal to 5"

def test_predict_result_6():
    """Test the model to see if it can accuretly predict an image of hand holding up 6"""
    img_path = "test_images/6/Sign 6 (181).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 6
    assert 6 == prediction, "Prediction should be equal to 6"

def test_predict_result_7():
    """Test the model to see if it can accuretly predict an image of hand holding up 7"""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 7
    assert 7 == prediction, "Prediction should be equal to 7"

def test_predict_result_8():
    """Test the model to see if it can accuretly predict an image of hand holding up 8"""
    img_path = "test_images/8/Sign 8 (3).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 8
    assert 8 == prediction, "Prediction should be equal to 8"

def test_predict_result_9():
    """Test the model to see if it can accuretly predict an image of hand holding up 9"""
    img_path = "test_images/9/Sign 9 (142).jpeg"
    processed_img = preprocess_img(img_path)

    # Make a prediction
    prediction = predict_result(processed_img)


    # Check that the prediction is equal to 9
    assert 9 == prediction, "Prediction should be equal to 9"

# Advanced Tests

def test_invalid_image_path():
    """Test preprocess_img with an invalid image path."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("invalid/path/to/image.jpeg")

def test_image_shape_on_prediction(model):
    """Test the prediction output shape."""
    img_path = "test_images/5/Sign 5 (86).jpeg"  # Ensure the path is correct
    processed_img = preprocess_img(img_path)

    # Ensure that the prediction output is an integer
    prediction = predict_result(processed_img)
    assert isinstance(prediction, (int, np.integer)), "The prediction should be an integer"

def test_model_predictions_consistency(model):
    """Test that predictions for the same input are consistent."""
    img_path = "test_images/7/Sign 7 (54).jpeg"
    processed_img = preprocess_img(img_path)

    # Make multiple predictions
    predictions = [predict_result(processed_img) for _ in range(5)]

    # Check that all predictions are the same
    assert all(p == predictions[0] for p in predictions), "Predictions for the same input should be consistent"
