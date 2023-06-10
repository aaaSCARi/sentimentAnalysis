import torch
from flask import Flask, request, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Approach 1: Using VADER Model:

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")

        # Perform sentiment analysis using SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(inp)

        # Get the sentiment label based on the negative score
        if scores["neg"] != 0:
            sentiment_label = "Negative☹☹"
        else:
            sentiment_label = "Positive😀😀"

        # If we want to classify into three labels:
        #
        # # Determine the sentiment label based on the scores
        # if scores["compound"] >= 0.05:
        #     sentiment_label = "Positive😀😀"
        # elif scores["compound"] <= -0.05:
        #     sentiment_label = "Negative☹☹"
        # else:
        #     sentiment_label = "Neutral"

        # Get the sentiment score for each category
        neg_score = scores["neg"]
        neu_score = scores["neu"]
        pos_score = scores["pos"]

        message = f"{sentiment_label}\nSentiment Scores:\nNegative={neg_score:.3f}\nNeutral={neu_score:.3f}\nPositive={pos_score:.3f}"

        return render_template('home.html', message=message)

    return render_template('home.html')



@app.route('/welcome', methods=["GET"])
def welcome():
    return render_template("welcome.html")


if __name__ == "__main__":
        app.run(debug=True)

# # Approach 2: Using RoBERTa Model:
#
# # Load the RoBERTa model and tokenizer
# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#
# @app.route('/', methods=["GET", "POST"])
# def main():
#     if request.method == "POST":
#         inp = request.form.get("inp")
#
#         # Preprocess the input text
#         inputs = tokenizer.encode_plus(inp, add_special_tokens=True, truncation=True, padding='max_length',
#                                        max_length=512, return_tensors='pt')
#
#         # Make sentiment prediction using the RoBERTa model
#         logits = model(**inputs).logits
#         predicted_class = int(torch.argmax(logits, dim=1))
#
#         # Get the RoBERTa scores
#         roberta_scores = torch.softmax(logits, dim=1).tolist()[0]
#
#         if predicted_class == 0:
#             sentiment_label = "Negative☹☹"
#         elif predicted_class == 1:
#             sentiment_label = "Neutral"
#         else:
#             sentiment_label = "Positive😀😀"
#
#         message = f"{sentiment_label} Sentiment: Negative={roberta_scores[0]:.3f}, Neutral={roberta_scores[1]:.3f}, Positive={roberta_scores[2]:.3f}"
#
#         return render_template('home.html', message=message)
#
#     return render_template('home.html')
