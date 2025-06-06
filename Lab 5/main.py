import os
# Workaround for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Используемое устройство: {device}')

class AgeDataset(Dataset):
    """Кастомный датасет для классификации возрастных групп"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['age_18_20', 'age_21_23', 'age_24_25']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # Загрузка путей к изображениям и меток
        try:
            print(f"Загрузка данных из {root_dir}...")
            for class_name in self.class_names:
                class_path = os.path.join(root_dir, class_name)
                if not os.path.exists(class_path):
                    print(f"Предупреждение: Директория {class_path} не найдена")
                    continue
                img_count = 0
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            Image.open(img_path).convert('RGB')
                            self.images.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])
                            img_count += 1
                        except Exception as e:
                            print(f"Предупреждение: Не удалось открыть изображение {img_path}: {str(e)}")
                print(f"Найдено {img_count} изображений в {class_path}")
            if not self.images:
                raise ValueError(f"Не найдено ни одного изображения в {root_dir}. Проверьте структуру директорий и наличие файлов .png/.jpg/.jpeg.")
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {str(e)}")
            raise

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка при открытии изображения {img_path}: {str(e)}")
            raise
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Трансформации для обучения и валидации
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(num_classes=3, pretrained=True):
    """Создание предобученной модели ResNet18"""
    try:
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model
    except Exception as e:
        print(f"Ошибка при создании модели: {str(e)}")
        raise

def train_model(model, train_loader, val_loader, num_epochs=5):
    """Обучение модели"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_train / total_train:.2f}%'
            })

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct_val / total_val:.2f}%'
                })

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100. * correct_train / total_train
        epoch_val_acc = 100. * correct_val / total_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 60)

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, class_names):
    """Оценка модели на тестовых данных"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Тестирование'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Отчет по классификации:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    cm = confusion_matrix(all_labels, all_predictions)

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Матрица путаницы')
        plt.ylabel('Истинные классы')
        plt.xlabel('Предсказанные классы')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Матрица путаницы сохранена как 'confusion_matrix.png'")
        plt.close()
    except Exception as e:
        print(f"Ошибка при построении матрицы путаницы: {str(e)}")

    return all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Визуализация процесса обучения"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Потери во время обучения')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(train_accuracies, label='Train Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Точность во время обучения')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Точность (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("График обучения сохранен как 'training_history.png'")
        plt.close()
    except Exception as e:
        print(f"Ошибка при построении графика обучения: {str(e)}")
        print(f"Train Losses: {train_losses[:5]} ...")
        print(f"Val Losses: {val_losses[:5]} ...")
        print(f"Train Accuracies: {train_accuracies[:5]} ...")
        print(f"Val Accuracies: {val_accuracies[:5]} ...")

# Основной код для выполнения
if __name__ == "__main__":
    # Параметры
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    NUM_CLASSES = 3

    # Пути к данным (используем абсолютный путь для надежности)
    BASE_DIR = r'C:\Users\Artem\OneDrive\Рабочий стол\DeepNeuro2\deepneuro\lab 5\dataset'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR = os.path.join(BASE_DIR, 'val')
    TEST_DIR = os.path.join(BASE_DIR, 'test')

    # Проверка существования директорий
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.exists(dir_path):
            print(f"Ошибка: Директория {dir_path} не существует")
            sys.exit(1)

    # Создание датасетов
    print("Загрузка данных...")
    try:
        train_dataset = AgeDataset(TRAIN_DIR, transform=train_transform)
        val_dataset = AgeDataset(VAL_DIR, transform=val_transform)
        test_dataset = AgeDataset(TEST_DIR, transform=val_transform)
    except ValueError as e:
        print(f"Ошибка при создании датасета: {str(e)}")
        sys.exit(1)

    print(f"Размер обучающего набора: {len(train_dataset)}")
    print(f"Размер валидационного набора: {len(val_dataset)}")
    print(f"Размер тестового набора: {len(test_dataset)}")

    # Создание загрузчиков данных
    try:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Ошибка при создании DataLoader: {str(e)}")
        sys.exit(1)

    # Создание и подготовка модели
    print("Создание модели...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Обучение модели
    print("Начинаем обучение...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS
    )

    # Визуализация результатов обучения
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Оценка на тестовых данных
    print("Оценка модели...")
    class_names = ['18-20 лет', '21-23 года', '24-25 лет']
    predictions, labels = evaluate_model(trained_model, test_loader, class_names)

    # Сохранение модели
    torch.save(trained_model.state_dict(), 'age_classification_model.pth')
    print("Модель сохранена как 'age_classification_model.pth'")

    print("Обучение завершено!")