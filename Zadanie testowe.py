def main():
    name = input("Podaj imię: ").strip()
    if not name:
        name = "World"
    print(f"Hello, {name}!")

if __name__ == "__main__":
    main()
