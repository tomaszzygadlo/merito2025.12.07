import os
import transformers
import torch as _torch
from transformers import pipeline

def ensure_model_local(model_name: str, model_dir: str) -> str:
    """
    Prosta pomocnicza funkcja:
    - jeśli model już znajduje się w model_dir (folder istnieje i nie jest pusty), zwraca model_dir,
    - w przeciwnym razie zwraca nazwę modelu (użyje pobrania z Hugging Face przez transformers).
    Możesz tutaj zamienić implementację na snapshot_download(...) jeśli chcesz wymusić pobranie.
    """
    try:
        if os.path.isdir(model_dir) and os.listdir(model_dir):
            return model_dir
    except Exception:
        pass
    return model_name

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_dir = os.path.join(os.path.dirname(__file__), "models", "distilbert-sst2")

local_model_path = ensure_model_local(model_name, model_dir)

# Usuń jawne ustawienie framework i pozwól transformers autodetekcji.
try:
    classifier = pipeline(
        "sentiment-analysis",
        model=local_model_path,
        tokenizer=local_model_path
    )
except Exception as e:
    # Fallback: spróbuj użyć nazwy modelu (transformers pobierze model automatycznie)
    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name
    )

print(classifier("James bond is a woman!"))
print(classifier("James bond is a woman! I hate woman"))
print(classifier("The movie was average but acceptable. Not good, not bad."))
print(classifier("It is terrible. I want my money back!"))