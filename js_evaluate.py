import numpy as np
from sklearn.metrics import recall_score, f1_score


def evaluate_model(model, test_user, test_item, test_rating):
    user = test_user
    item = test_item
    rating = test_rating
    predictions = model.predict([user, item], batch_size=256)
    if isinstance(predictions, list):
        predictions = predictions[0]
    predictions = predictions.flatten()
    predictions = np.round(predictions).astype(int)
    evals = getRecall(rating, predictions)
    return evals


def getRecall(rating, predictions):
    # k = np.random.rand(len(rating))
    # k = np.round(k).astype(int)
    recall = recall_score(rating, predictions)
    a = f1_score(rating, predictions)

    return (recall, a)
