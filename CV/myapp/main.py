import numpy as np

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from .model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    print('=== НАЧАЛО ОБРАБОТКИ НА СЕРВЕРЕ ===')
    print('1. Получена строка от frontend:', image[:50] + '...')
    print('   Длина строки:', len(image))
    
    # Преобразование строки в массив
    image = np.array(list(map(int, image[1:-1].split(','))))
    print('2. Преобразовано в numpy массив:', image[:10], '...')
    print('   Размер массива:', image.shape)
    print('   Мин/Макс значений:', image.min(), '/', image.max())
    print('   Количество ненулевых пикселей:', np.count_nonzero(image))
    
    pred = model.predict(image)
    print('3. Предсказание модели:', pred)
    print('   ASCII код:', ord(pred))
    print('=== КОНЕЦ ОБРАБОТКИ НА СЕРВЕРЕ ===')
    
    return {'prediction': pred}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
