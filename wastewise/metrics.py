import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


def temperature_scale(probs, temperature):
    logits = np.log(np.clip(probs, 1e-9, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate(y, probs):
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda t: log_loss(y, temperature_scale(probs, t), labels=np.arange(probs.shape[1])), bounds=(.2, 5), method='bounded')
    return float(result.x)


def report(y, probs, labels, threshold=.8):
    y = np.asarray(y)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    accepted = confidence >= threshold
    ece = 0.
    for lo in np.arange(0, 1, .1):
        mask = (confidence >= lo) & (confidence < lo + .1 if lo < .9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - confidence[mask].mean())
    n = len(y)
    acc = float(accuracy_score(y, pred))
    z = 1.96
    center = (acc + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n, accuracy=acc, accuracy_wilson_95=[float(center-half), float(center+half)],
                macro_f1=float(f1_score(y, pred, labels=range(len(labels)), average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                log_loss=float(log_loss(y, probs, labels=range(len(labels)))),
                expected_calibration_error=float(ece), threshold=threshold,
                coverage=float(accepted.mean()), accepted_accuracy=float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None,
                confusion_matrix=confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
                per_class=classification_report(y, pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0))
