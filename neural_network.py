"""
Модуль для загрузки и использования предобученных нейросетей
для распознавания параметров оптических вихрей Лагерра-Гаусса
"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

# Количество классов для параметров
N_CLASSES_N = 7  # n от 1 до 7
N_CLASSES_M = 7  # m от 2 до 8

class AlexNetLG(nn.Module):
    """
    AlexNet архитектура для распознавания параметров оптических вихрей
    Используется для загрузки state_dict обученных моделей
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.output_n = nn.Linear(4096, N_CLASSES_N)
        self.output_m = nn.Linear(4096, N_CLASSES_M)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.output_n(x), self.output_m(x)

class VortexRecognizer:
    """
    Класс для распознавания оптических вихрей с использованием предобученных моделей
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Загрузка предобученной модели из файла PyTorch
        
        Args:
            model_path: путь к файлу модели (.pth или .pt)
            
        Returns:
            bool: True если загрузка успешна, False в противном случае
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Обработка разных форматов сохранения
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Загрузка state_dict в архитектуру AlexNetLG
                    self.model = AlexNetLG().to(self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    # Если это просто dict без специальных ключей, считаем это state_dict
                    try:
                        self.model = AlexNetLG().to(self.device)
                        self.model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"Ошибка загрузки state_dict в архитектуру AlexNetLG: {e}")
                        print("Архитектура вашей модели может отличаться от AlexNetLG.")
                        return False
            else:
                # Загружен полный объект модели
                self.model = checkpoint
            
            # Перевод в режим оценки
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            print(f"Модель успешно загружена из {model_path}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def load_image(self, image_source):
        """
        Загрузка изображения из файла или из массива numpy
        
        Args:
            image_source: путь к файлу (str) или numpy array
            
        Returns:
            numpy.ndarray: загруженное изображение или None при ошибке
        """
        try:
            if isinstance(image_source, str):
                # Загрузка из файла
                image = Image.open(image_source)
                return np.array(image)
            elif isinstance(image_source, np.ndarray):
                # Возврат массива numpy
                return image_source
            elif isinstance(image_source, Image.Image):
                # Конвертация PIL Image в numpy array
                return np.array(image_source)
            else:
                print("Неподдерживаемый формат изображения")
                return None
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None
    
    def recognize(self, image_input):
        """
        Распознавание параметров оптического вихря
        
        Args:
            image_input: изображение в формате numpy array или путь к файлу
            
        Returns:
            dict: {'n': радиальный индекс, 'm': азимутальный индекс, 'TC': топологический заряд}
                  или None при ошибке
        """
        if self.model is None:
            print("Модель не загружена")
            return None
        
        # Загрузка изображения
        image = self.load_image(image_input)
        if image is None:
            return None
        
        try:
            # Подготовка тензора (соответствует предобработке при обучении)
            if len(image.shape) == 2:
                # Градации серого - конвертируем в RGB
                image = np.stack([image] * 3, axis=-1)
            
            # Убедимся, что изображение в RGB формате
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB - конвертируем в формат PyTorch (C, H, W)
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
            else:
                print("Неподдерживаемый формат изображения. Ожидается RGB.")
                return None
            
            # Масштабирование до 227x227 (как при обучении)
            from torchvision import transforms
            resize_transform = transforms.Resize((227, 227))
            image_tensor = resize_transform(image_tensor)
            
            # Нормализация 0-1 (как transforms.ToTensor())
            image_tensor = image_tensor / 255.0
            
            # Добавление размерности батча
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Прямой проход через модель
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    output = self.model(image_tensor)
                else:
                    output = self.model(image_tensor)
                
                # Обработка выхода модели
                if isinstance(output, tuple) and len(output) == 2:
                    n_pred, m_pred = output
                    n_class = torch.argmax(n_pred, dim=1).item()
                    m_class = torch.argmax(m_pred, dim=1).item()
                    
                    # Конвертация классов в реальные значения
                    # n: классы 0-6 -> значения 1-7
                    n_value = n_class + 1
                    # m: классы 0-6 -> значения 2-8
                    m_value = m_class + 2
                    
                    # Вычисление топологического заряда
                    tc_value = m_value - n_value
                    
                    return {
                        'n': n_value,
                        'm': m_value,
                        'TC': tc_value
                    }
                else:
                    # Если модель возвращает другой формат
                    print("Неподдерживаемый формат выхода модели")
                    return None
                    
        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return None
    
    def recognize_batch(self, image_paths):
        """
        Пакетное распознавание изображений
        
        Args:
            image_paths: список путей к изображениям
            
        Returns:
            list: список результатов распознавания
        """
        results = []
        for path in image_paths:
            result = self.recognize(path)
            if result:
                result['filename'] = os.path.basename(path)
                results.append(result)
        return results
