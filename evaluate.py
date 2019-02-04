import pandas as pd


def evaluate_model(model, testRatings, testNegatives, K):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Args:
        model: the model to be evaluated
        testRatings: rating dataframe for test
        testNegatives: negative lists for each user in test set
        K: top K evaluation
    Return: 
        score of each test rating. (Hit_Ratio, NDCG)
    """
    hits, ndcgs = [], []
    for idx, row in testRatings.iterrows():
        (hr, ndcg) = eval_one_rating(idx, row)
        hits.append(hr)
        ndcgs.append(ndcg)


def eval_one_rating(idx, row):
    