import numpy as np

# 1. Analiza podstawowych metryk listy cen produktów
# Losowanie 10 cen z przedziału od 20 do 99
ceny_produktow = np.random.uniform(20, 99, 10)

# Konwersja listy na tablicę NumPy dla efektywnych obliczeń
ceny_np = np.array(ceny_produktow)

print("--- Analiza Statystyczna Cen Produktów ---")
print(f"Oryginalne ceny: {ceny_np}")

# Obliczenie średniej, mediany i odchylenia standardowego
srednia_cena = np.mean(ceny_np)
mediana_cen = np.median(ceny_np)
odchylenie_std = np.std(ceny_np)

print(f"Średnia cena: {srednia_cena:.2f}")
print(f"Mediana cen: {mediana_cen:.2f}")
print(f"Odchylenie standardowe: {odchylenie_std:.2f}\n")


# 2. Stworzenie funkcji do normalizacji danych (skalowanie do zakresu 0-1)
def normalizuj_dane(dane):
    """
    Normalizuje dane w tablicy NumPy do zakresu od 0 do 1.
    """
    min_val = np.min(dane)
    max_val = np.max(dane)
    # Zabezpieczenie przed dzieleniem przez zero, gdy wszystkie elementy są takie same
    if max_val - min_val > 0:
        return (dane - min_val) / (max_val - min_val)
    else:
        return np.zeros(dane.shape)

# 3. Przetestowanie funkcji normalizującej
print("--- Normalizacja Danych ---")

# Normalizacja cen przy założeniu, że maksymalna możliwa cena to 100
# W tym przypadku normalizacja to po prostu podzielenie przez 100
znormalizowane_ceny = ceny_np / 100.0
print("Ceny po normalizacji (względem max 100):")
print(np.round(znormalizowane_ceny, 2))
print(f"Min po normalizacji: {np.min(znormalizowane_ceny):.2f}, Max po normalizacji: {np.max(znormalizowane_ceny):.2f}\n")

# Test na innym zestawie danych z użyciem oryginalnej funkcji
print("--- Dodatkowy Test Normalizacji (funkcja normalizuj_dane) ---")
dane_testowe = np.array([-5, 0, 10, 15, 20])
print(f"Oryginalne dane testowe: {dane_testowe}")

znormalizowane_dane_testowe = normalizuj_dane(dane_testowe)
print("Dane testowe po normalizacji:")
print(znormalizowane_dane_testowe)
print(f"Min po normalizacji: {np.min(znormalizowane_dane_testowe):.1f}, Max po normalizacji: {np.max(znormalizowane_dane_testowe):.1f}")

