BANKING_KEYWORDS = {
    "bank",
    "banking",
    "account",
    "card",
    "credit",
    "debit",
    "payment",
    "transfer",
    "transaction",
    "cash",
    "withdraw",
    "withdrawal",
    "deposit",
    "balance",
    "money",
    "refund",
    "charge",
    "fee",
    "top up",
    "top-up",
    "cash withdrawal",
    "pin",
    "iban",
    "currency",
    "exchange",
    "verification",
    "verify",
    "identity",
    "loan",
    "savings",
    "statement",
    "merchant",
    "atm",
    "wallet"
}


def is_banking_query(text):
    text = text.lower()

    return any(keyword in text for keyword in BANKING_KEYWORDS)