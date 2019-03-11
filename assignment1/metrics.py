def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for i in range(prediction.shape[0]):
        p, r = prediction[i], ground_truth[i]
        if p == r:
            if p:
                TP +=1
            else:
                TN +=1
        else:
            if p:
                FP += 1
            else:
                FN += 1
    
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / prediction.shape[0]
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        pass
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = 0
    num_test = prediction.shape[0]
    for j in range(num_test):
        if prediction[j] == ground_truth[j]:
            accuracy += 1
            
    accuracy /= num_test
    return accuracy
