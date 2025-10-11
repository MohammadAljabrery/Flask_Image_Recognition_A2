# pointless commit to test for something
# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Prediction route that processes uploaded images.

    Returns:
        str: Rendered HTML template with prediction results or error message
    """
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))
        return render_template("result.html", err="Invalid request method")
    except Exception as error:
        error_message = "File cannot be processed."
        print(f"Error occurred: {error}")
        return render_template("result.html", err=error_message)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
