from rouge import Rouge

_rouge = Rouge()

def rouge(predictions, targets):
    results = _rouge.get_scores(predictions, targets, avg=True)
    return results  # { 'rouge-1': {...}, 'rouge-2': {...}, ... }
