import re

from spellchecker import SpellChecker


spell = SpellChecker()

BANKING_TERMS = {
    "atm",
    "iban",
    "upi",
    "pin",
    "topup",
    "cashback",
    "cryptocurrency",
    "cryptocurrencies",
    "fintech"
}


def find_spelling_errors(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    words = [word for word in words if word not in BANKING_TERMS]

    unknown_words = spell.unknown(words)
    corrections = {}

    for word in unknown_words:
        correction = spell.correction(word)

        if correction and correction != word:
            corrections[word] = correction

    return corrections