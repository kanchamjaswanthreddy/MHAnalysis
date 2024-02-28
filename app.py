import os  # Import the os module
from flask import Flask, render_template, request
from transformers import pipeline
import csv
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize text classification pipeline
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Define the directory where prediction files will be stored
PREDICTION_DIR = '/Users/jaswanthreddykancham/Desktop/Projects/MH Analysis/Predictions'

def save_predictions_to_csv(predictions, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['label', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row

        # Write predictions directly to the CSV file
        for label, score in zip(predictions['label'], predictions['score']):
            writer.writerow({'label': label, 'score': score})
    # Print the filename
    print(f"Predictions saved to {filename}")

# Route to handle chatbot input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input text from the form
        input_text = request.form['input_text']

        # Perform analysis using the classifier
        model_outputs = classifier([input_text])

        # Process model outputs
        predictions = {'label': [], 'score': []}
        for output in model_outputs:
            for each in output:
                predictions['label'].append(each['label'])
                predictions['score'].append(each['score'])

        # Generate a unique filename with timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        # Modify the filename to include the desired directory path
        filename = os.path.join(PREDICTION_DIR, f"predictions_{timestamp}.csv")

        # Save predictions to CSV file
        save_predictions_to_csv(predictions, filename)

        # Return a success message
        return render_template('success.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
