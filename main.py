from flask import Flask
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, top_k=None)

@app.route('/')
def hello_world():
    results = classifier("Ananiya is Bullshit!!!")
    # for result in results[0]:
    #     print(result['label']
    # label_scores = {}
    # for item in results[0]:
    #     label_scores[item['label']] = item['score']
    # label_ranks = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)

    # for rank, (label, score) in enumerate(label_ranks, start=1):
    #     print(f"{label} => {rank}")

    highest = 0

    label_to_number = {
        'joy': 1,
        'surprise': 2,
        'neutral': 3,
        'anger': 4,
        'disgust': 5,
        'sadness': 6,
        'fear': 7
    }
    for result in results[0]:
        label = result['label']
        score = result['score']
        number = label_to_number[label]
        print(f"{label}: {number}, score: {score}")
        if number > highest:
            highest = number
        break
    print(highest)
    return 'Hello'

if __name__ == '__main__':
    app.run(port=8080)

# {'label': 'joy', 'score': 0.9225417971611023}
# {'label': 'surprise', 'score': 0.036415379494428635}
# {'label': 'neutral', 'score': 0.030083000659942627}
# {'label': 'anger', 'score': 0.004329684190452099}
# {'label': 'disgust', 'score': 0.003188240109011531}
# {'label': 'sadness', 'score': 0.002023096662014723}
# {'label': 'fear', 'score': 0.001418727682903409}