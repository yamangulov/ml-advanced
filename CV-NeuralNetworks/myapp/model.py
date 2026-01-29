import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class CNNModel(nn.Module):
    def __init__(self, num_classes=47):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Model:
    """
    Класс для загрузки и использования обученной модели распознавания символов.

    Методы:
    - __init__(): Загружает модель из файла model.ckpt
    - predict(image): Выполняет предсказание для изображения

    Формат входных данных:
    - image: str в формате "[1,2,3,...]" - строка с пикселями изображения

    Формат выходных данных:
    - str: Распознанный символ
    """

    def __init__(self):
        """Инициализация модели и загрузка весов."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNModel(num_classes=47)

        # Загрузка модели с абсолютным путем
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model.ckpt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Загрузка mapping для преобразования индексов в символы
        mapping_path = os.path.join(os.path.dirname(__file__), '..', 'emnist-balanced-mapping.txt')
        with open(mapping_path, 'r') as f:
            lines = f.readlines()

        self.mapping = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                char_code = int(parts[1])
                self.mapping[idx] = chr(char_code)

    def predict(self, image_str):
        """
        Выполняет предсказание для изображения.

        Args:
            image_str (str): Строка с пикселями в формате "[1,2,3,...]"

        Returns:
            str: Распознанный символ
        """
        # Преобразование строки в тензор (28, 28)
        image = torch.tensor(list(map(int, image_str[1:-1].split(',')))).reshape((28, 28))

        # Преобразование изображения с правильной нормализацией
        image = image.unsqueeze(0).unsqueeze(0).float()  # Добавляем batch и channel размерности
        image = image / 255.0  # Нормализация к [0, 1]
        image = (image - 0.1307) / 0.3081  # Стандартизация с параметрами EMNIST

        # Предсказание
        with torch.no_grad():
            output = self.model(image.to(self.device))
            _, predicted = torch.max(output, 1)
            predicted_idx = predicted.item()

        # Преобразование индекса в символ
        if predicted_idx in self.mapping:
            return self.mapping[predicted_idx]
        else:
            return '?'
