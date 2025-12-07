import pandas as pd
import random
from faker import Faker

# Generowanie losowych danych klientów (5000 rekordów)
NUM_RECORDS = 5000


# jeśli chcesz zachować oryginalne listy jako fallback, one pozostają w pliku
fake = Faker('pl_PL')
dane = {
    'Imię': [fake.first_name() for _ in range(NUM_RECORDS)],
    'Wiek': [fake.random_int(min=18, max=80) for _ in range(NUM_RECORDS)],
    'Miasto': [fake.city() for _ in range(NUM_RECORDS)],
    'Pensja': [fake.random_int(min=3000, max=15000) for _ in range(NUM_RECORDS)]
}
print("Generowanie danych: użyto Faker (pl_PL).")

df = pd.DataFrame(dane)

# Wypisz krótki podgląd i podstawowe informacje
print(df.head(10))
print(f"Liczba rekordów: {len(df)}")

# Oblicz średnią wieku klientów i wypisz wynik
avg_age = df['Wiek'].mean()
print(f"Średni wiek klientów: {avg_age:.2f}")

# Zapis do pliku (xlsx jeśli dostępny, w przeciwnym razie CSV)
output_path = r"dane.xlsx"
try:
    import openpyxl  # sprawdź, czy silnik do xlsx jest dostępny
    df.to_excel(output_path, index=False)
    print(f"Zapisano do pliku: {output_path}")
except ModuleNotFoundError:
    csv_path = r"dane.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("Moduł 'openpyxl' nie jest zainstalowany. Zapisano jako CSV:")
    print(f"  {csv_path}")
    print("Aby zapisać do XLSX zainstaluj openpyxl, np.: pip install openpyxl")
