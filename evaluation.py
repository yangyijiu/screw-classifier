import torch
from sklearn.metrics import roc_auc_score
# Functions related to model evaluation will be added here.
def f1_score(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives + 1e-10)  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives + 1e-10)  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall + 1e-10)  # Compute F1 score


def evaluate(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            true_positives += (labels * predicted).sum().float()
            false_positives += ((1 - labels) * predicted).sum().float()
            false_negatives += (labels * (1 - predicted)).sum().float()

    print(true_positives, false_positives, false_negatives)
    return f1_score(true_positives, false_positives, false_negatives)




def compute_auc_roc(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Get softmax probabilities
            softmax_probs = torch.nn.functional.softmax(model(inputs), dim=1)

            # Save the probabilities for class 1 (positive class)
            pos_probs = softmax_probs[:, 1].cpu().numpy()

            predicted_probs.extend(pos_probs)
            true_labels.extend(labels.cpu().numpy())

        return roc_auc_score(true_labels, predicted_probs)
