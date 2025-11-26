from nltk.translate.bleu_score import sentence_bleu

def bleu(predictions, targets):
    scores = []
    for pred, tgt in zip(predictions, targets):
        scores.append(sentence_bleu([tgt.split()], pred.split()))
    return sum(scores) / len(scores) if scores else 0
