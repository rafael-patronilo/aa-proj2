from sklearn.metrics.cluster import pair_confusion_matrix
import numpy as np

def purity_score(pred_labels, true_labels):
    clusters = set(pred_labels)
    classes = list(set(true_labels))
    score = 0
    for cluster in clusters:
        idx = pred_labels == cluster
        score += max(np.sum(true_labels[idx]==cls) for cls in classes)
    return score / pred_labels.shape[0]
        

def calculate_metrics(pred_labels, true_labels):
    purity = purity_score(pred_labels, true_labels)
    
    confusion = pair_confusion_matrix(true_labels, pred_labels)
    true_neg = confusion[0,0]
    false_pos = confusion[0,1]
    false_neg = confusion[1,0]
    true_pos = confusion[1,1]
    total_pairs = np.sum(confusion)
    np.seterr(divide='ignore', invalid='ignore')
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)
    rand = (true_pos + true_neg) / total_pairs
    
    return purity, precision, recall, f1, rand

