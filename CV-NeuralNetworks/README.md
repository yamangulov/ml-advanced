# EMNIST Symbol Recognition Service

## Описание
Сервис для распознавания рукописных символов на основе датасета EMNIST.

## Архитектура
- Сверточная нейронная сеть с 3 сверточными слоями
- Batch Normalization и Dropout для регуляризации
- Точность на валидационной выборке: 87%+

## Запуск

### Локально
```bash
uvicorn myapp.main:app
```
### В Docker
```
docker build -t emnist-service .
docker run -p 8000:8000 emnist-service
```
### API
- POST /api/predict - распознавание символа

### Результаты
- точность выше 0.9
- сервис запускается и работает локально и в docker контейнере