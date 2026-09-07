# Spam Classifier

My second AI/ML project. It reads a text message and predicts whether it's **spam** or **ham** (not spam).

## What it does

1. Takes a text message as input
2. Converts it into numbers using TF-IDF
3. Feeds it into a Logistic Regression model
4. Predicts spam or ham

## Why I built this

Project #2 in my AI engineering journey — built solo this time, applying what I learned from my first project (sentiment classifier).

## Dataset

SMS Spam Collection dataset — ~5,500 real text messages labeled spam/ham.

## Tech used

- Python
- scikit-learn
- pandas

## How to run

```bash
pip install -r requirements.txt
python train.py
python predict.py "Congratulations! You've won a free prize, click here to claim"
```

## Status

🚧 Work in progress — learning as I build.

## What's next

- Try other models and compare accuracy
- Explore precision/recall (spam detection cares more about false positives)
- Move on to building a RAG chatbot