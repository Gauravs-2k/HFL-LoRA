import re

def normalize_text(text):
    """Standard cleaning for text/policy answers."""
    return str(text).strip().lower()

def normalize_code(code):
    """
    Aggressive code normalization to handle format differences.
    1. Removes comments and docstrings.
    2. Removes whitespace.
    3. Lowercases everything.
    """
    code = str(code)
    # Remove docstrings ("""...""")
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    # Remove comments (#...)
    code = re.sub(r'#.*', '', code)
    # Remove all whitespace
    code = re.sub(r'\s+', '', code)
    return code.lower()

def extract_numbers(text):
    """Extracts numbers for Finance/HR data."""
    text_clean = str(text).replace(',', '')
    matches = re.findall(r"-?\d+\.?\d*", text_clean)
    valid_nums = set()
    for m in matches:
        try:
            if m.strip() != '.':
                valid_nums.add(float(m))
        except ValueError:
            continue
    return valid_nums

def is_correct(pred, target):
    """
    Smart matcher that routes to the right logic based on content.
    """
    # 1. Code Match (If target looks like Python code)
    if "def " in target or "class " in target or "return " in target:
        # Check if functional logic is identical (ignoring formatting)
        if normalize_code(pred) == normalize_code(target):
            return True
            
    # 2. Number Match (For Finance/HR)
    target_nums = extract_numbers(target)
    if target_nums:
        pred_nums = extract_numbers(pred)
        if target_nums.issubset(pred_nums):
            return True
            
    # 3. Fallback: Text Match (For Policy QA)
    # Use token overlap for long answers
    pred_tokens = set(normalize_text(pred).split())
    target_tokens = set(normalize_text(target).split())
    
    # If >70% of target keywords are in prediction, pass it
    if len(target_tokens) > 5:
        overlap = len(target_tokens.intersection(pred_tokens))
        if overlap / len(target_tokens) > 0.7:
            return True
            
    # 4. Last Resort: Substring
    if normalize_text(target) in normalize_text(pred):
        return True
        
    return False

def accuracy(predictions, targets):
    if not targets: return 0.0
    correct_count = 0
    for p, t in zip(predictions, targets):
        if is_correct(p, t):
            correct_count += 1
    return correct_count / len(targets)
