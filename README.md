🚀 Uruchomienie Systemu Detekcji Anomalii
Jeśli chcesz przetwarzać obrazy krok po kroku (Normalizacja, Filtrowanie, a następnie Detekcja Anomalii), postępuj zgodnie z poniższą sekwencją komend.

1. Normalizacja Danych
Ten krok prawdopodobnie przygotowuje obrazy do dalszego przetwarzania lub trenowania, standaryzując ich wartości (np. intensywność pikseli).

Uruchom skrypt do normalizacji danych:

Bash

python data_normalization.py
2. Zastosowanie Filtra
Po normalizacji należy zastosować odpowiedni filtr (np. w celu redukcji szumów lub uwydatnienia krawędzi) za pomocą skryptu apply_filter.py.

Uruchom skrypt do zastosowania filtru:

Bash

python apply_filter.py
