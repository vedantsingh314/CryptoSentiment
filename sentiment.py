from transformers import pipeline
from utils import detect_coin

# 🔥 Lazy loading (IMPORTANT FIX)
classifier = None

def get_model():
    global classifier
    if classifier is None:
        print("Loading sentiment model...")
        classifier = pipeline("sentiment-analysis")
    return classifier


def analyze(news_list):
    results = {}

    model = get_model()  # ✅ load once when needed

    for text in news_list:
        coin = detect_coin(text)
        print(text)

        if not coin:
            continue

        # ✅ use model instead of classifier
        output = model(text)[0]

        sentiment = output["label"]
        confidence = output["score"]

        if coin not in results:
            results[coin] = []

        results[coin].append({
            "sentiment": sentiment,
            "confidence": confidence
        })

    return results


def aggregate(results):
    final = {}

    for coin, entries in results.items():
        pos = 0
        neg = 0

        for item in entries:
            if item["sentiment"] == "POSITIVE":
                pos += 1
            else:
                neg += 1

        if pos > neg:
            sentiment = "Bullish 📈"
        elif neg > pos:
            sentiment = "Bearish 📉"
        else:
            sentiment = "Neutral 😐"

        avg_confidence = sum(item["confidence"] for item in entries) / len(entries)

        final[coin] = {
            "sentiment": sentiment,
            "confidence": round(avg_confidence, 2)
        }

    return final