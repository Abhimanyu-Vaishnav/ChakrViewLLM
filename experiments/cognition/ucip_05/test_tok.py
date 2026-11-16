import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
from scripts.run_m3_evaluations import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(Path('data/experiments/vocab_4096'))

test_cases = [
    # English Basic
    'Hello world.', 'The sky is blue.', 'Good morning!', 'What is a computer?',
    'A central processing unit executes software instructions.',
    'I have 5 apples and give 2 away.',
    'Quick brown fox jumps over the lazy dog.',
    'Special characters: !@#$%^&*()_+-=[]{}|;:,.?/~',
    # Hindi Devanagari
    'नमस्ते दुनिया।', 'आप कैसे हैं?', 'मैं ठीक हूँ, धन्यवाद।',
    'कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है।',
    'भारत एक महान और सुंदर देश है।',
    'शिक्षा मनुष्य के जीवन को सुंदर बनाती है।',
    'हिंदी भाषा देवनागरी लिपि में लिखी जाती है।',
    # Roman Hindi
    'Aap kaise hain?', 'Main theek hoon, dhanyawaad.',
    'Bhai, machine learning ko simple language mein samjhao.',
    'Computer ka CPU instructions execute karta hai.',
    'Aaj mausam bahut accha hai.',
    # Hinglish
    'ChakrView indigenous local neural engine hai jo CPU par efficiently run karta hai.',
    'Python programming seekhna kaafi aasan aur useful hai.',
    'Yeh code efficiently kaam kar raha hai.',
    # Arithmetic & Numbers
    '1 + 1 = 2', '10 - 4 = 6', '5 * 5 = 25', '100 / 10 = 10',
    # Formatting
    'Line1\nLine2\nLine3'
]

passed = 0
for tc in test_cases:
    enc = tok.encode(tc, add_bos=False, add_eos=False)
    dec = tok.decode(enc)
    if dec == tc:
        passed += 1
    else:
        print(f"MISMATCH:\n  Orig: {repr(tc)}\n  Dec:  {repr(dec)}")

print(f"Total tested: {len(test_cases)}, Exact matches: {passed}/{len(test_cases)}")
