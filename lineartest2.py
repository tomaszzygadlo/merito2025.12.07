import numpy as np
import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Krok 1: Wczytanie i przygotowanie danych z repozytorium GitHub
# Używamy biblioteki pandas do wczytania danych z pliku CSV.
url = 'https://raw.githubusercontent.com/paulsamuel-w-e/E-commerce-Customer-Behaviour-Dataset/main/E-commerce.csv'

def calculate_total_purchase(history_str):
    """Funkcja do parsowania kolumny 'Purchase History' i sumowania cen."""
    if not isinstance(history_str, str):
        return 0.0
    # Używamy wyrażenia regularnego do znalezienia wszystkich cen w stringu
    prices = re.findall(r'["\']Price["\']\s*:\s*(\d+\.?\d*)', history_str)
    return sum(float(p) for p in prices)

try:
    df = pd.read_csv(url)

    # Przetwarzanie danych w celu uzyskania sumy zakupów
    df['Total Purchase Amount'] = df['Purchase History'].apply(calculate_total_purchase)

    # Wybieramy interesujące nas kolumny: 'Time on Site' jako cecha (X)
    # i nowo utworzoną 'Total Purchase Amount' jako zmienna docelowa (y).
    # Usuwamy wiersze z brakującymi wartościami i zerową kwotą zakupu.
    df_clean = df[['Time on Site', 'Total Purchase Amount']].dropna()
    df_clean = df_clean[df_clean['Total Purchase Amount'] > 0]

    # Krok 2: Podział danych na zbiór uczący i testowy
    # Zbiór testowy to 10 losowych rekordów, reszta to zbiór uczący.
    test_set = df_clean.sample(n=10, random_state=42)
    train_set = df_clean.drop(test_set.index)

    X_train = train_set[['Time on Site']].values
    y_train = train_set[['Total Purchase Amount']].values
    X_test = test_set[['Time on Site']].values
    y_test = test_set[['Total Purchase Amount']].values

except Exception as e:
    print(f"Nie udało się wczytać lub przetworzyć danych. Błąd: {e}")
    print("Używanie danych syntetycznych jako alternatywy.")
    # W przypadku problemu z pobraniem danych, wracamy do danych syntetycznych
    np.random.seed(0)
    X = 30 * np.random.rand(100, 1)
    y = 15 + 5 * X + np.random.randn(100, 1) * 20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Krok 3: Trenowanie i ocena modelu
model = LinearRegression()

# Krok 3a: Trenowanie modelu regresji liniowej
model.fit(X_train, y_train)

# Krok 3b: Przewidywanie wartości na zbiorze testowym
y_pred = model.predict(X_test)

# Porównanie cen dla zbioru testowego
print("\nPorównanie cen dla zbioru testowego (10 rekordów):")
print("-----------------------------------------------------------------")
print(f"{'Cena realna':>15} | {'Cena prognozowana':>20} | {'Różnica absolutna':>20}")
print("-----------------------------------------------------------------")
for real, pred in zip(y_test, y_pred):
    real_price = real[0]
    pred_price = pred[0]
    diff = abs(real_price - pred_price)
    print(f"{real_price:>14.2f} PLN | {pred_price:>19.2f} PLN | {diff:>19.2f} PLN")
print("-----------------------------------------------------------------")


# Krok 4: Ocena skuteczności modelu
print("\nOcena modelu na danych testowych:")
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Współczynnik R-kwadrat (R^2): {r2:.2f}")
print(f"Błąd średniokwadratowy (MSE): {mse:.2f}")
print(f"Średnia różnica ceny (MAE): {mae:.2f} PLN")

# Krok 5: Parametry modelu
print(f"\nParametry modelu:")
print(f"Współczynnik (nachylenie): {model.coef_[0][0]:.2f}")
print(f"Wyraz wolny (przecięcie z osią Y): {model.intercept_[0]:.2f}")


# Krok 6: Przewidywanie dla nowej wartości
# Jaka będzie przewidywana kwota zakupu, jeśli klient spędzi na stronie 10 minut?
time_spent = [[10]]
pred = model.predict(time_spent)
print(f"\nPrzewidywana kwota zakupu dla {time_spent[0][0]} minut na stronie: {pred[0][0]:.2f} PLN")

# Krok 7: Wizualizacja wyników
# Używamy pełnego zbioru danych do wizualizacji, aby pokazać ogólny trend
X_full = df_clean[['Time on Site']].values
y_full_pred = model.predict(X_full)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Dane uczące')
plt.scatter(X_test, y_test, color='blue', edgecolor='k', s=100, label='Dane testowe (10 rekordów)')
plt.plot(X_full, y_full_pred, color='red', linewidth=2, label='Linia regresji')
plt.title('Regresja liniowa: Czas na stronie vs. Kwota zakupu')
plt.xlabel('Czas spędzony na stronie')
plt.ylabel('Kwota zakupu (PLN)')
plt.legend()
plt.grid(True)
plt.show()
