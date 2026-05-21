from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

mapping_dict = {
    'spam_status': {
        0: 'Ham (Safe)',
        1: 'Spam (Danger)'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        sms_text = request.form['sms_message']

        prediction = model.predict([sms_text])[0]

        predicted_status = mapping_dict['spam_status'][prediction]

        return render_template(
            'index.html',
            prediction_text=f"Result: {predicted_status}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)