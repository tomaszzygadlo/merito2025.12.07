import sys
import os
import traceback
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def detect_framework():
    try:
        import torch  # type: ignore
        return "pt"
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore
        # upewnij się, że tf.__version__ >= "2.0"
        return "tf"
    except Exception:
        pass
    try:
        import jax  # type: ignore
        return "flax"
    except Exception:
        pass
    return None

fw = detect_framework()
if fw is None:
    print("Brak zainstalowanego PyTorch, TensorFlow >= 2.0 lub Flax.")
    print(r"Aby naprawić (przykład dla venv i CPU):")
    print(r".\.venv\Scripts\activate")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# lokalna ścieżka projektu -> models/cardiffnlp-twitter-xlm-roberta-base-sentiment
BASE_DIR = os.path.dirname(__file__)
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "cardiffnlp-twitter-xlm-roberta-base-sentiment")

def ensure_local_model(model_name: str, local_dir: str):
    """
    Pobiera model/tokenizer z HF i zapisuje lokalnie. W razie brakujących zależności (tiktoken/protobuf/sentencepiece)
    zwraca (None, None) oraz wypisuje instrukcję naprawczą zamiast przerywać cały proces wyjątkiem.
    """
    os.makedirs(local_dir, exist_ok=True)
    need_download = not bool(os.listdir(local_dir))

    def try_load(name_or_path, use_fast=None, local_files_only=False):
        kwargs = {}
        if use_fast is not None:
            kwargs["use_fast"] = use_fast
        if local_files_only:
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(name_or_path, local_files_only=local_files_only)
        return tokenizer, model

    missing_instructions = (
        "Brakuje wymaganych bibliotek. Aby spróbować naprawić, uruchom:\n"
        "  pip install protobuf sentencepiece tiktoken huggingface-hub transformers\n"
        "Po instalacji zrestartuj skrypt. Jeśli działasz na Windows i widzisz ostrzeanie o symlinkach,\n"
        "włącz Developer Mode lub uruchom Pythona jako administrator."
    )

    if need_download:
        try:
            tokenizer, model = try_load(model_name, use_fast=None, local_files_only=False)
        except ImportError as ie:
            print("ImportError podczas ładowania tokenizera/modelu:", ie)
            print(missing_instructions)
            return None, None
        except ValueError as ve:
            # np. konwersja slow->fast nie powiodła się (tiktoken/sentencepiece)
            print("ValueError podczas ładowania tokenizera/modelu:", ve)
            print(missing_instructions)
            return None, None
        except Exception as e1:
            warnings.warn("Pierwsza próba załadowania tokenizera/modelu nie powiodła się: " + str(e1))
            # spróbuj fallback bez fast tokenizer
            try:
                tokenizer, model = try_load(model_name, use_fast=False, local_files_only=False)
            except ImportError as ie2:
                print("ImportError podczas ładowania (fallback):", ie2)
                print(missing_instructions)
                return None, None
            except Exception as e2:
                # spróbuj załadować z lokalnego katalogu jeśli coś tam jest
                if os.listdir(local_dir):
                    try:
                        tokenizer, model = try_load(local_dir, use_fast=False, local_files_only=True)
                    except Exception:
                        print("Nie udało się załadować ani pobrać modelu/tokenizera.")
                        print(missing_instructions)
                        traceback.print_exc()
                        return None, None
                else:
                    print("Nie udało się pobrać modelu/tokenizera oraz brak lokalnych plików.")
                    print(missing_instructions)
                    traceback.print_exc()
                    return None, None
        # zapisz pobrany model/tokenizer lokalnie (bez przerywania przy błędzie zapisu)
        try:
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
        except Exception as save_exc:
            warnings.warn("Zapis modelu/tokenizera na dysku nie powiódł się: " + str(save_exc))
        return tokenizer, model
    else:
        # katalog nie jest pusty — spróbuj załadować lokalnie (najpierw fast, potem slow)
        try:
            tokenizer, model = try_load(local_dir, use_fast=None, local_files_only=True)
            return tokenizer, model
        except ImportError as ie_local:
            print("ImportError przy ładowaniu lokalnego modelu/tokenizera:", ie_local)
            print(missing_instructions)
            return None, None
        except Exception:
            try:
                tokenizer, model = try_load(local_dir, use_fast=False, local_files_only=True)
                return tokenizer, model
            except Exception as e:
                print("Nie udało się załadować lokalnego modelu/tokenizera z:", local_dir)
                print(missing_instructions)
                traceback.print_exc()
                return None, None

# przygotuj pipeline (return_all_scores=True aby otrzymać pełny rozkład prawdopodobieństw)
tokenizer, model = ensure_local_model(MODEL_NAME, LOCAL_MODEL_DIR)
if tokenizer is None or model is None:
    print("Model/tokenizer nie zostały załadowane. Pipeline nie zostanie utworzony.")
    sentiment_pipeline = None
else:
    try:
        # Używamy return_all_scores=True, aby otrzymać wyniki dla wszystkich klas
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
    except Exception as e:
        print("Utworzenie pipeline nie powiodło się:", e)
        print("Spróbuj zainstalować brakujące zależności: pip install tiktoken protobuf sentencepiece")
        sentiment_pipeline = None

def analyze(text: str):
    """
    Analiza sentymentu (model wielojęzyczny - działa z polskim tekstem).
    Jeśli pipeline nie jest dostępny, rzuca RuntimeError z instrukcjami naprawy.
    """
    if sentiment_pipeline is None:
        raise RuntimeError(
            "Pipeline sentymentu nie jest dostępny. Zainstaluj brakujące zależności:\n"
            "  pip install protobuf sentencepiece tiktoken huggingface-hub transformers\n"
            "Potem uruchom skrypt ponownie."
        )
    text = text.strip()
    return sentiment_pipeline(text)

# przykład użycia
if __name__ == "__main__":
    texts_to_analyze = [
        "James Bond jest kobietą!",
        "Ten film jest okropny, nie podobał mi się.",
        "Film był przeciętny, ale akceptowalny. Ani dobry, ani zły.",
        "To jest okropne. Chcę zwrotu pieniędzy!",
        "Ten film był wspaniały",
        "To jest jakieś paskudztwo. Totalne gówno! Tylko debil z tego skorzysta!"
    ]

    # Poprawna mapa etykiet dla modelu 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
    label_map = {
        "negative": "Negatywny",
        "neutral": "Neutralny",
        "positive": "Pozytywny"
    }

    print("--- Analiza sentymentu z użyciem modelu 'cardiffnlp/twitter-xlm-roberta-base-sentiment' ---\n")
    for text in texts_to_analyze:
        try:
            if sentiment_pipeline:
                # Wynik jest teraz listą w liście, np. [[{'label':...}, ...]]
                full_result = analyze(text)[0]

                # Znajdź etykietę z najwyższym wynikiem
                best_result = max(full_result, key=lambda x: x['score'])
                label = best_result['label']
                score = best_result['score']
                sentiment_class = label_map.get(label, "Nieznany")

                print(f"Tekst: \"{text}\"")
                print(f"  -> Zinterpretowany sentyment: {sentiment_class}, Pewność: {score:.2f}")
                print(f"  -> Pełna odpowiedź modelu: {full_result}\n")
            else:
                print("Pipeline nie jest dostępny, analiza przerwana.")
                break
        except RuntimeError as e:
            print(f"Błąd podczas analizy tekstu: \"{text}\"")
            print(e)
            break # Przerwij pętlę, jeśli pipeline nie działa
