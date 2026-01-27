import os
import pickle
import numpy as np

class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание
        '''
        # Преобразование изображения в вектор и нормализация
        x_vector = x.reshape(1, -1) / 255.0
        # Предсказание
        pred_label = self.model.predict(x_vector)[0]
        # Преобразование лейбла в символ
        return chr(pred_label)