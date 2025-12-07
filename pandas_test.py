import pandas as pd

dane = {
    'Imię': ['Zenon', 'Otylia', 'Krzysztof', 'Piotr'],
    'Wiek': [25, 30, 28, 35],
    'Miasto': ['Warszawa', 'Wąchock', 'Gdańsk', 'Wrocław'],
    'Pensja': [5000, 6000, 5500, 7000]
}

df = pd.DataFrame(dane)
print(df)

output_path = r"dane.xlsx"
try:
    import openpyxl  # sprawdź, czy silnik do xlsx jest dostępny
    df.to_excel(output_path, index=False)
    print(f"Zapisano do pliku: {output_path}")
except ModuleNotFoundError:
    csv_path = r"C:\src\Merito TorunLodz 07.10.2025\dane.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("Moduł 'openpyxl' nie jest zainstalowany. Zapisano jako CSV:")
    print(f"  {csv_path}")
    print("Aby zapisać do XLSX zainstaluj openpyxl, np.: pip install openpyxl")
