import re

def normalize_text(text):
    """Cleans text for comparison (lowercase, strip whitespace)."""
    return str(text).strip().lower()

def extract_numbers(text):
    """
    Extracts specific numbers from text to handle formatting differences.
    Example: "The total is 8,520.25" -> {8520.25}
    Example: "0.51%" -> {0.51}
    """
    # Remove commas (e.g., 1,000 -> 1000) so numbers match regardless of format
    text_clean = str(text).replace(',', '')
    
    # Find all integer or decimal numbers
    matches = re.findall(r"-?\d+\.?\d*", text_clean)
    
    # Convert to floats for value-based comparison (0.510 == 0.51)
    valid_nums = set()
    for m in matches:
        try:
            if m.strip() != '.':
                valid_nums.add(float(m))
        except ValueError:
            continue
    return valid_nums

def is_correct(pred, target):
    """Checks if the prediction contains the target answer."""
    # 1. Fast Path: Exact String Match
    if normalize_text(pred) == normalize_text(target):
        return True

    # 2. Number Match Strategy (Critical for Finance/HR data)
    target_nums = extract_numbers(target)
    pred_nums = extract_numbers(pred)
    
    if target_nums:
        # If the target has numbers (e.g. "0.51"), they MUST be in the prediction
        # Using issubset() allows the model to say "The ratio is 0.51"
        return target_nums.issubset(pred_nums)
    
    # 3. Fallback: Substring Match (for text answers like "Yes/No")
    # e.g. Target="Yes", Pred="Yes, absolutely."
    if normalize_text(target) in normalize_text(pred):
        return True
        
    return False

def accuracy(predictions, targets):
    """
    Calculates accuracy allowing for chatty responses and formatting differences.
    """
    if not targets:
        return 0.0

    correct_count = 0
    for p, t in zip(predictions, targets):
        if is_correct(p, t):
            correct_count += 1
            
    return correct_count / len(targets)
