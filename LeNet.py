import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Определение архитектуры LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Сверточные слои
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 1 канал (grayscale), 6 фильтров
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # Подвыборка
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 6 каналов, 16 фильтров
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # Подвыборка
        # Полносвязные слои
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 * 5 * 5 = 400 входных нейронов, 120 выходных
        self.fc2 = nn.Linear(120, 84)  # 120 входных нейронов, 84 выходных
        self.fc3 = nn.Linear(84, 10)  # 84 входных нейронов, 10 выходных (классы цифр)

    def forward(self, x):
        # Проход через сверточные слои
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # Выравнивание тензора для полносвязных слоев
        x = x.view(-1, 16 * 5 * 5)
        # Проход через полносвязные слои
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Загрузка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование изображений в тензоры
    transforms.Normalize((0.1307,), (0.3081,))  # Нормализация
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Инициализация модели, функции потерь и оптимизатора
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Обнуление градиентов
            optimizer.zero_grad()
            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Эпоха {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Тестирование модели
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Точность на тестовых данных: {100 * correct / total}%")

# Запуск обучения и тестирования
train(model, train_loader, criterion, optimizer, epochs=10)
test(model, test_loader)













