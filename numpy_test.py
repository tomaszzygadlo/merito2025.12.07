import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a + b)  # [5 7 9]

# Dodane: generowanie danych i obliczenia statystyczne
rng = np.random.default_rng()
arr = rng.integers(1, 101, size=20)  # 20 liczb z zakresu 1-100 (inclusive)

mean_val = arr.mean()
median_val = np.median(arr)
std_val = arr.std()  # odchylenie standardowe (population)

greater_than_mean = arr[arr > mean_val]

sorted_asc = np.sort(arr)
sorted_desc = sorted_asc[::-1]

matrix_4x5 = arr.reshape(4, 5)

print("Tablica 20 losowych liczb:", arr)
print("Średnia:", mean_val)
print("Mediana:", median_val)
print("Odchylenie standardowe:", std_val)
print("Liczby większe niż średnia:", greater_than_mean)
print("Posortowane rosnąco:", sorted_asc)
print("Posortowane malejąco:", sorted_desc)
print("Macierz 4x5:\n", matrix_4x5)
