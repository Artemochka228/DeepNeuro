import os
# Workaround for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('dataset_simple.csv')
    print("Данные:")
    print(df.head())
    print(f"Размер данных: {df.shape}")
except FileNotFoundError:
    print("Ошибка: Файл 'dataset_simple.csv' не найден.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Ошибка: Файл 'dataset_simple.csv' пуст или поврежден.")
    exit(1)

# Check for missing or invalid data
if df.isnull().any().any():
    print("Ошибка: В данных есть пропущенные значения.")
    print(df.isnull().sum())
    exit(1)

# Extract features (age) and target (income)
X = torch.tensor(df.iloc[:, [0]].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).reshape(-1, 1)

print(f"\nРазмерность признаков X: {X.shape}")
print(f"Размерность меток y: {y.shape}")
print(f"Первые 5 примеров X:\n{X[:5]}")
print(f"Первые 5 меток y: {y[:5].flatten()}")

# Normalize the data
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
if torch.any(X_std == 0):
    print("Ошибка: Стандартное отклонение для X равно 0.")
    exit(1)
X_normalized = (X - X_mean) / X_std

y_mean = y.mean(dim=0)
y_std = y.std(dim=0)
if torch.any(y_std == 0):
    print("Ошибка: Стандартное отклонение для y равно 0.")
    exit(1)
y_normalized = (y - y_mean) / y_std

print(f"\nНормализованные данные X (первые 5 примеров):\n{X_normalized[:5]}")
print(f"Нормализованные данные y (первые 5 примеров):\n{y_normalized[:5].flatten()}")

# Define the neural network for regression
class NNet_Regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet_Regression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, X):
        return self.layers(X)

# Network parameters
inputSize = X.shape[1]
hiddenSizes = 16
outputSize = 1

print(f"\nПараметры сети:")
print(f"Входной слой: {inputSize} нейронов")
print(f"Скрытый слой: {hiddenSizes} нейронов")
print(f"Выходной слой: {outputSize} нейрон")

# Initialize the network
net = NNet_Regression(inputSize, hiddenSizes, outputSize)

print(f"\nПараметры сети:")
for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")

# Loss function and optimizer
lossFn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

print(f"\nНачинаем обучение...")
epochs = 200

# Training loop
for i in range(epochs):
    pred = net(X_normalized)
    loss = lossFn(pred, y_normalized)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 20 == 0:
        print(f'Ошибка на {i + 1} итерации: {loss.item():.4f}')

print("\nРЕЗУЛЬТАТЫ ОБУЧЕНИЯ")

# Evaluate and make predictions
with torch.no_grad():
    pred_normalized = net(X_normalized)
    pred = pred_normalized * y_std + y_mean
    mae = torch.mean(torch.abs(pred - y))
    print(f"Средняя абсолютная ошибка (MAE): ${mae.item():.2f}")

# Example prediction for age 40
new_age = torch.tensor([[40.0]], dtype=torch.float32)
new_age_normalized = (new_age - X_mean) / X_std
with torch.no_grad():
    pred_income_normalized = net(new_age_normalized)
    pred_income = pred_income_normalized * y_std + y_mean
    print(f"Предсказанный доход для возраста 40: ${pred_income.item():.2f}")

# Visualize results
try:
    plt.figure(figsize=(8, 6))
    X_np = X.numpy().flatten()
    y_np = y.numpy().flatten()
    pred_np = pred.numpy().flatten()

    sorted_indices = np.argsort(X_np)
    X_sorted = X_np[sorted_indices]
    y_sorted = y_np[sorted_indices]
    pred_sorted = pred_np[sorted_indices]

    plt.scatter(X_np, y_np, label='Фактический доход', alpha=0.5, color='blue')
    plt.scatter(X_np, pred_np, label='Предсказанный доход', alpha=0.5, color='red')
    plt.plot(X_sorted, pred_sorted, color='red', linestyle='--', label='Линия предсказания')
    plt.xlabel('Возраст')
    plt.ylabel('Доход')
    plt.title('Возраст vs Доход: Фактический и Предсказанный')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('income_plot.png')
    print("График сохранен как 'income_plot.png'")
except Exception as e:
    print(f"Ошибка при построении графика: {str(e)}")
    print("Данные для графика:")
    print(f"X (возраст): {X_np[:5]} ...")
    print(f"y (фактический доход): {y_np[:5]} ...")
    print(f"pred (предсказанный доход): {pred_np[:5]} ...")

print(f"\nОбучение завершено!")