import pandas as pd
import numpy as np
import yfinance as yf

# Descargar precios desde Yahoo Finance
tickers = ["IVV", "ILF", "SOXX"]  # S&P 500, LATAM, semiconductores
vector_w = [0.5, 0.3, 0.2]  # Lista de pesos del portafolio
vector_w = np.array(vector_w).reshape(-1, 1)  # Vector columna de pesos
df_prices = yf.download(tickers, start="2024-01-01", end="2026-09-30")["Close"]

# Calcular retornos diarios
df_ret = df_prices.pct_change().dropna()

# Calcular matriz de varianza-covarianza
df_cov = df_ret.cov()

# Calcular varianza del portafolio
w = vector_w
w_t = w.T
sigma = df_cov.values

print("w", w.shape)
print("w", w_t.shape)
print("sigma", sigma.shape)

varianza = np.dot(sigma, w)
varianza = np.dot(w_t, varianza)
varianza = varianza[0][0]

# Calcular desviación estándar
desv = varianza**(1/2)
volatilidad_anual = desv * 252**(1/2)
print(f"La volatilidad del portafolio es: {volatilidad_anual*100:.04}%")

#calcular media del portfolio
vector_ret = df_ret.mean()
vector_ret = vector_ret.values
vector_ret = vector_ret.reshape(-1,1)
print(vector_ret.shape)
mu = np.dot(w_t, vector_ret) [0][0]
mu_anual = mu * 252
print(f"La anual del portfolio es: {mu_anual*100:.04}%")


#calcular value at risk (VaR) al 95% de confianza

var_95 = mu_anual - 1.64 * volatilidad_anual
print(f"La VaR del portfolio es: {var_95*100:.04}%")



##con una prob de 95% la pérdida diara es 9.51% del portfolio