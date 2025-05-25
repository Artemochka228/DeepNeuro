import torch
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df.head())

y = df.iloc[:, 4].values

import numpy as np
y_numpy = np.where(y == "Iris-setosa", 1, -1)
y = torch.tensor(y_numpy, dtype=torch.int32)

X = torch.tensor(df.iloc[:, [0, 1, 2]].values, dtype=torch.float32)

print(f"Размерность данных X: {X.shape}")
print(f"Размерность ответов y: {y.shape}")


def neuron(w, x):
    if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0:
        predict = 1
    else:
        predict = -1
    return predict


w = torch.tensor([0, 0.1, 0.4, 0.2], dtype=torch.float32)
print(f"Предсказание нейрона для первого примера: {neuron(w, X[0])}")


w = torch.rand(4, dtype=torch.float32)
eta = 0.01
w_iter = []

print("Начинаем обучение...")
for j in range(X.shape[0]):
    xi = X[j]
    target = y[j]
    predict = neuron(w, xi)
    error = target - predict
    w[1:] += eta * error * xi
    w[0] += eta * error

    if j % 10 == 0:
        w_iter.append(w.clone())


print(f"Финальные веса: {w}")

sum_err = 0
correct_predictions = 0
total_predictions = len(X)

for j in range(len(X)):
    xi = X[j]
    target = y[j]
    predict = neuron(w, xi)
    if predict != target:
        sum_err += 1
    else:
        correct_predictions += 1

print(f"Всего ошибок: {sum_err}")
print(f"Точность: {correct_predictions / total_predictions:.3f} ({correct_predictions}/{total_predictions})")
print(f"Количество эпох обучения: {len(w_iter)}")

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', label='Iris-setosa')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x', label='Iris-versicolor')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Классификация ирисов (показаны только первые 2 признака)')
plt.legend()
plt.grid(True)
plt.show()