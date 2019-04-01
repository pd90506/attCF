import numpy as np
import math


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
    # Convert sample items to a big matrix, each row represents a user
    # the first column is the real test sample, and other columns are
    # negative test samples
    testRatings_array = testRatings.iloc[:, 1].values.reshape((-1, 1))
    testNegatives_array = np.asarray(testNegatives.iloc[:, 1].tolist())
    testSamples = np.concatenate((testRatings_array, testNegatives_array), axis=1)

    # Create user and item input samples
    shape = testSamples.shape
    testSamples = testSamples.reshape(-1, 1)
    userSamples = np.asarray(range(shape[0])).repeat(shape[1])

    # Make prediction with given samples, and reshape it to matrix form
    # the first column is the prediction of test sample, others are
    # of the negative samples
    predictions = model.predict([userSamples, testSamples], batch_size=256)
    if isinstance(predictions, list):
        predictions = predictions[0]
    predictions = predictions.reshape(shape)

    hits = getHitRatio(predictions, K)
    ndcgs = getNDCG(predictions, K)

    return (hits, ndcgs)


def getHitRatio(predictions, K=10):
    """calculate the hit ratio
    the first column of the predictions matrix is the real test samples
    """
    out = np.apply_along_axis(hitFirstK, axis=1, arr=predictions)
    return out


def hitFirstK(array, K=10):
    """check if the first element ranks in first K
    """
    # argsort find the index of K largest elements
    topK = array.argsort()[-K:]
    # find if 0 is in the largest K
    out = 1 if (0 in topK) else 0
    return out


def getNDCG(predictions, K=10):
    # TODO!!!!
    out = np.apply_along_axis(gainFirstK, axis=1, arr=predictions)
    return out


def gainFirstK(array, K=10):
    """calculate the cumulative gain for each real sample
    """
    topK = np.flip(array.argsort()[-K:])
    a, = np.where(topK == 0)
    if len(a) > 0:
        return math.log(2.0) / math.log(a[0] + 2.0)
    else:
        return 0.0




    
