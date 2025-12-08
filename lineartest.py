import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Krok 1: Wczytanie danych z nowego źródła
try:
    url = 'https://raw.githubusercontent.com/am-tropin/poland-apartment-prices/main/house-prices-in-poland/Houses.csv'
    df_all = pd.read_csv(url, on_bad_lines='skip', encoding='latin1')

    # Czyszczenie i przygotowanie głównego zbioru danych
    df_all = df_all[['city', 'sq', 'price']].rename(columns={'city': 'Miasto', 'sq': 'Powierzchnia', 'price': 'Cena'})
    df_all['Powierzchnia'] = pd.to_numeric(df_all['Powierzchnia'], errors='coerce')
    df_all['Cena'] = pd.to_numeric(df_all['Cena'], errors='coerce')
    df_all.dropna(inplace=True)

    # Wybieramy miasta z największą liczbą ofert do analizy (w tym Wrocław)
    cities_to_analyze = df_all['Miasto'].value_counts().nlargest(5).index

except Exception as e:
    print(f"Nie udało się wczytać głównego pliku danych. Błąd: {e}")
    cities_to_analyze = []


# Przygotowanie do wspólnej wizualizacji
plt.figure(figsize=(14, 8))
colors = plt.cm.get_cmap('tab10', len(cities_to_analyze))

# Pętla po miastach
for i, city in enumerate(cities_to_analyze):
    print(f"\n{'='*20} Analiza dla: {city.upper()} {'='*20}")
    try:
        # Krok 2: Przygotowanie danych dla konkretnego miasta
        df_city = df_all[df_all['Miasto'] == city]

        df_clean = df_city[(df_city['Cena'] > 1000) & (df_city['Powierzchnia'] > 10) & (df_city['Powierzchnia'] <= 250)]

        if len(df_clean) < 20:
            print(f"Niewystarczająca ilość danych dla {city} po czyszczeniu.")
            continue

        # Krok 3: Podział danych
        test_set = df_clean.sample(n=10, random_state=42)
        train_set = df_clean.drop(test_set.index)

        X_train = train_set[['Powierzchnia']].values
        y_train = train_set[['Cena']].values
        X_test = test_set[['Powierzchnia']].values
        y_test = test_set[['Cena']].values

        # Krok 4: Trenowanie i ocena modelu
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Krok 5: Wyświetlanie wyników
        print("\nPorównanie cen dla zbioru testowego (10 rekordów):")
        print("-----------------------------------------------------------------")
        print(f"{'Cena realna':>15} | {'Cena prognozowana':>20} | {'Różnica absolutna':>20}")
        print("-----------------------------------------------------------------")
        for real, pred in zip(y_test, y_pred):
            real_price, pred_price = real[0], pred[0]
            diff = abs(real_price - pred_price)
            print(f"{real_price:>14.2f} PLN | {pred_price:>19.2f} PLN | {diff:>19.2f} PLN")
        print("-----------------------------------------------------------------")

        print("\nOcena modelu na danych testowych:")
        print(f"Współczynnik R-kwadrat (R^2): {r2_score(y_test, y_pred):.2f}")
        print(f"Średnia różnica ceny (MAE): {mean_absolute_error(y_test, y_pred):.2f} PLN")

        print(f"\nParametry modelu:")
        print(f"Przewidywana cena za m²: {model.coef_[0][0]:.2f} PLN")
        print(f"Bazowa cena nieruchomości: {model.intercept_[0]:.2f} PLN")

        area = [[50]]
        pred_price = model.predict(area)
        print(f"\nPrzewidywana cena dla nieruchomości o powierzchni {area[0][0]} m²: {pred_price[0][0]:.2f} PLN")

        # Krok 6: Dodawanie danych do wspólnego wykresu
        X_full = df_clean[['Powierzchnia']].values
        y_full_pred = model.predict(X_full)

        # Sortowanie wartości dla płynnej linii regresji
        sort_axis = np.argsort(X_full.ravel())
        X_full_sorted = X_full[sort_axis]
        y_full_pred_sorted = y_full_pred[sort_axis]

        # Rysowanie punktów i linii regresji dla bieżącego miasta
        plt.scatter(X_train, y_train, color=colors(i), alpha=0.2, s=20, label=f'Dane - {city}')
        plt.plot(X_full_sorted, y_full_pred_sorted, color=colors(i), linewidth=3, label=f'Regresja - {city}')

    except Exception as e:
        print(f"Nie udało się przetworzyć danych dla {city}. Błąd: {e}")

# Finalizacja wspólnego wykresu
plt.title('Regresja: Powierzchnia vs. Cena w dużych miastach Polski')
plt.xlabel('Powierzchnia (m²)')
plt.ylabel('Cena (PLN)')
plt.legend()
plt.grid(True)
plt.show()
