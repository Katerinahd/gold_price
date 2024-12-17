import pandas as pd
#priprava dat pro linearni regresi
# Načtení dat z Excel souboru
df = pd.read_csv(r'C:\Users\kater\Desktop\gold_prices.csv')
  # Zde změňte cestu k vašemu souboru

# Zobrazení prvních pár řádků pro kontrolu
print(df.head())


df["Date"]=pd.to_datetime(df["Date"])
print(df["Date"].head())

df.set_index('Date', inplace=True)
print(df.head())

window_size = 5  # Počet předchozích dní, které chceme použít

# Vytvořte nové sloupce pro cenu 'Close' za posledních 5 dní (lagy)
for i in range(1, window_size + 1):
    df[f'Close_lag_{i}'] = df['Close'].shift(i)

# Odstraníme řádky s NaN (první 5 dní nebude mít všechny hodnoty pro lagy)
df = df.dropna()

# Zobrazte výsledky pro kontrolu
print(df.head())

# Vstupy - všechny sloupce, které použijeme pro predikci ceny
X = df[['Open', 'High', 'Low', 'Volume'] + [f'Close_lag_{i}' for i in range(1, window_size + 1)]]

# Výstupy - sloupec 'Close' bude naše cílová proměnná
y = df['Close']

# Zkontrolujte, zda jsou vstupy a výstupy správně vytvořeny
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

# Rozdělení dat na trénovací a testovací sadu (80% trénování, 20% testování)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Zkontrolujte velikost trénovací a testovací sady
print(f"Trénovací sada: {X_train.shape[0]} řádků")
print(f"Testovací sada: {X_test.shape[0]} řádků")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Vytvoření instance modelu
model = LinearRegression()

# Trénink modelu na trénovacích datech
model.fit(X_train, y_train)
# Predikce hodnot na testovací sadě
y_pred = model.predict(X_test)
# Výpočet střední kvadratické chyby (MSE)
mse = mean_squared_error(y_test, y_pred)

# Výpočet koeficientu determinace (R²)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")


import matplotlib.pyplot as plt

# Scatter plot skutečných vs. předpovězených hodnot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Skutečné hodnoty")
plt.ylabel("Předpovězené hodnoty")
plt.title("Skutečné vs. předpovězené hodnoty")
plt.show()

# Histogram chyb (zbytky)
errors = y_test - y_pred
plt.hist(errors, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel("Chyba (zbytek)")
plt.ylabel("Frekvence")
plt.title("Histogram chyb")
plt.show()

import joblib

# Uložení modelu do souboru
joblib.dump(model, 'linear_regression_gold_model.pkl')

# Načtení modelu (v budoucnu)
loaded_model = joblib.load('linear_regression_gold_model.pkl')

# Příklad nových dat (musí mít stejný formát jako X)
new_data = pd.DataFrame({
    'Open': [1950],
    'High': [1965],
    'Low': [1945],
    'Volume': [20000],
    'Close_lag_1': [1948],
    'Close_lag_2': [1947],
    'Close_lag_3': [1946],
    'Close_lag_4': [1945],
    'Close_lag_5': [1944],
})

# Předpověď
new_prediction = model.predict(new_data)
print(f"Předpovězená cena zlata: {new_prediction[0]}")

import pandas as pd
import numpy as np
from datetime import timedelta

# Výchozí bod: poslední známá data z testovací sady
last_known_data = X_test.iloc[-1].copy()

# Počet dní k predikci (5 let = 5 * 365 dní)
future_days = 5 * 365

# Seznam pro ukládání predikcí
future_predictions = []

# Smyčka pro iterativní predikci
for _ in range(future_days):
    # Predikce na základě aktuálních dat
    predicted_price = model.predict([last_known_data])[0]
    future_predictions.append(predicted_price)

    # Aktualizace lagů
    for lag in range(window_size, 1, -1):
        last_known_data[f'Close_lag_{lag}'] = last_known_data[f'Close_lag_{lag - 1}']
    last_known_data['Close_lag_1'] = predicted_price

# Vytvoření DataFrame pro predikce
last_date = X_test.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
future_df.set_index('Date', inplace=True)
import matplotlib.pyplot as plt

# Spojení skutečných hodnot s budoucími predikcemi
all_data = pd.concat([df[['Close']], future_df.rename(columns={'Predicted_Close': 'Close'})])

# Vykreslení
plt.figure(figsize=(14, 8))
plt.plot(all_data.index, all_data['Close'], label='Ceny zlata (skutečné + predikované)', color='blue')
plt.axvline(x=X_test.index[-1], color='red', linestyle='--', label='Začátek predikce')
plt.xlabel('Datum')
plt.ylabel('Cena zlata')
plt.title('Predikce ceny zlata na následujících 5 let')
plt.legend()
plt.show()
