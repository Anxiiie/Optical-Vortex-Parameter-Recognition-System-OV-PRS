"""
Module for loading and using pre-trained neural networks for recognizing 
the Laguerre-Gaussian optical vortices parameters
"""

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

# Number of classes for parameters
N_CLASSES_N = 7  # n от 1 до 7
N_CLASSES_M = 7  # m от 2 до 8

class AlexNetLG(nn.Module):
    """
    AlexNet architecture for optical vortex parameter recognition
    Used to load the state_dict of pre-trained models (You can change to any other architecture)
    Use only state_dict save (Recommended by PyTorch developers)
    (Described here: http://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
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
    Class for optical vortex recognition using pre-trained models
    """
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Upload pre-trained model from PyTorch file
        
        Args:
            model_path: path to model file (.pth или .pt)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handling different saving formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Upload state_dict into AlexNetLG architecture
                    self.model = AlexNetLG().to(self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    # If it's just a dict with no special keys, consider it a state_dict
                    try:
                        self.model = AlexNetLG().to(self.device)
                        self.model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"Ошибка загрузки state_dict в архитектуру AlexNetLG: {e}")
                        print("Архитектура вашей модели может отличаться от AlexNetLG.")
                        return False
            else:
                # The complete model object has been loaded
                self.model = checkpoint
            
            # Switching to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            print(f"Модель успешно загружена из {model_path}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def load_image(self, image_source):
        """
        Loading an image from a file or numpy array
        
        Args:
            image_source: path to file (str) or numpy array
            
        Returns:
            numpy.ndarray: uploaded image or None on error
        """
        try:
            if isinstance(image_source, str):
                # Upload from file
                image = Image.open(image_source)
                return np.array(image)
            elif isinstance(image_source, np.ndarray):
                # Returning a numpy array
                return image_source
            elif isinstance(image_source, Image.Image):
                # Converting a PIL Image to a NumPy Array
                return np.array(image_source)
            else:
                print("Неподдерживаемый формат изображения")
                return None
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None
    
    def recognize(self, image_input):
        """
        Recognition of optical vortex parameters
        
        Args:
            image_input: image in numpy array format or file path
            
        Returns:
            dict: {'n': radial index, 'm': azimuthal index, 'TC': topological charge}
                    or None on error
        """
        if self.model is None:
            print("Модель не загружена")
            return None
        
        # Uploading an image
        image = self.load_image(image_input)
        if image is None:
            return None
        
        try:
            # Tensor preparation (corresponds to preprocessing during training)
            if len(image.shape) == 2:
                # Grayscale - Convert to RGB
                image = np.stack([image] * 3, axis=-1)
            
            # Let's make sure the image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB - convert to PyTorch format (C, H, W)
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
            else:
                print("Неподдерживаемый формат изображения. Ожидается RGB.")
                return None
            
            # Scaling to 227x227 (as in training)
            from torchvision import transforms
            resize_transform = transforms.Resize((227, 227))
            image_tensor = resize_transform(image_tensor)
            
            # Normalization 0-1 (like transforms.ToTensor())
            image_tensor = image_tensor / 255.0
            
            # Adding batch dimensions
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Direct pass through the model
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    output = self.model(image_tensor)
                else:
                    output = self.model(image_tensor)
                
                # Processing the model output
                if isinstance(output, tuple) and len(output) == 2:
                    n_pred, m_pred = output
                    n_class = torch.argmax(n_pred, dim=1).item()
                    m_class = torch.argmax(m_pred, dim=1).item()
                    
                    # Converting classes to real values
                    # n: classes 0-6 -> values ​​1-7
                    n_value = n_class + 1
                    # m: classes 0-6 -> values 2-8
                    m_value = m_class + 2
                    
                    # Calculation of topological charge
                    tc_value = m_value - n_value
                    
                    return {
                        'n': n_value,
                        'm': m_value,
                        'TC': tc_value
                    }
                else:
                    # If the model returns a different format
                    print("Неподдерживаемый формат выхода модели")
                    return None
                    
        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return None
    
    def recognize_batch(self, image_paths):
        """
        Batch image recognition
        
        Args:
            image_paths: list of paths to images
            
        Returns:
            list: list of recognition results
        """
        results = []
        for path in image_paths:
            result = self.recognize(path)
            if result:
                result['filename'] = os.path.basename(path)
                results.append(result)
        return results
