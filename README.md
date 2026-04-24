# Optical Vortex Parameter Recognition System (OV-PRS)

**[Русский](README_ru.md) | [中文](README_zh.md) | [日本語](README_ja.md)**

The Optical Vortex Parameter Recognition System (OV-PRS) is designed to solve the specialized task of recognizing parameters of Laguerre-Gaussian optical vortices from intensity images. The program uses machine learning methods to analyze input data obtained from a camera of an experimental setup and automatically determines LG-mode parameters, including radial and azimuthal indices. Based on the recognized parameters, the topological charge of the optical vortex can also be calculated.

The implementation is oriented towards operation under turbulent distortions that inevitably arise in real optical systems. The neural network model used is trained on data simulating various levels of turbulence, ensuring robust parameter recognition even with significant intensity profile distortions.

The software is created for a specific research task, but the solution architecture allows adaptation to other types of optical modes, different beam parameters, or different experimental conditions. If necessary, the system can be modified to user requirements: the set of recognized parameters can be changed, the model can be expanded, a new camera or data format can be integrated, and it can be trained for specific light propagation conditions.

## Features

- **Parameter Recognition**: Determination of radial index (n), azimuthal index (m), and topological charge (TC)
- **Operating Modes**:
  - Real-time image capture from web camera
  - Loading individual images from disk
  - Batch processing of multiple images
- **Export Results**: Save recognition results in CSV format
- **Web Interface**: Work through browser on localhost
- **Multi-language Support**: English, Russian, Chinese, Japanese

## System Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Flask 2.3+
- NumPy, Pillow, torchvision

**Note**: Due to the specialized nature of the application for optical vortex recognition, specific requirements are imposed on neural network models.

## Installation

1. Clone repository:
```bash
git clone https://github.com/Anxiiie/Optical-Vortex-Parameter-Recognition-System-OV-PRS.git
cd optical-vortex-recognizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application
```bash
python main.py
```

After starting, open your browser and navigate to: http://127.0.0.1:5000

### Operating Modes

1. **Load Model**:
   - Click "Load Model" and select a pre-trained model file (.pth or .pt)
   - The model will be loaded and ready for use

2. **Camera Operation**:
   - Click "Connect Camera"
   - The application will start capturing images from the web camera in real-time
   - Parameter recognition occurs automatically
   - Results are displayed in the right panel
   - Click "Stop Camera" to finish

3. **Image Loading**:
   - Click "Load Image" to select a single file
   - The image will be automatically processed
   - Recognition results will appear immediately after loading

4. **Batch Processing**:
   - Click "Batch Load" to select multiple images
   - All images will be processed sequentially
   - Results will be added to the history table

5. **View and Export Results**:
   - Click "Show Table" to view all results
   - Click "Export CSV" to save results to a file
   - Click "Clear Results" to delete all data

## Neural Network Model

### Model Requirements
- Format: PyTorch (.pth or .pt)
- Output: tuple of two tensors (predictions for n and m)
- Input: RGB image, 227x227 pixels

### Parameter Ranges
- **Radial Index (n)**: 7 classes (from 1 to 7)
- **Azimuthal Index (m)**: 7 classes (from 2 to 8)
- **Topological Charge (TC)**: calculated as TC = m - n (always >= 1)

## Model Preparation

The application supports loading pre-trained PyTorch models in state_dict format (weights dictionary).

### Model Saving Format

Saving only model weights (recommended by PyTorch developers):

```python
import torch

# Save state_dict
torch.save(model.state_dict(), 'model_state_dict.pth')

# Or with wrapper
torch.save({'model_state_dict': model.state_dict()}, 'model_state_dict.pth')
```

The application uses AlexNetLG architecture to load state_dict. Ensure your model architecture matches the application's AlexNetLG architecture.

## Results Format

Recognition results are saved in CSV format with the following columns:
- `filename`: Image file name
- `n`: Radial index
- `m`: Azimuthal index
- `TC`: Topological charge

## Troubleshooting

### Camera Issues
- Ensure the web camera is connected and working
- Check that other applications are not using the camera
- Restart the application if errors occur

### Model Loading Issues
- Ensure the model file has .pth or .pt format
- Check that the model is compatible with the PyTorch version
- Check console output for errors

### Recognition Issues
- Ensure the model is loaded correctly
- Check image quality
- Ensure the image format is supported

### Startup Errors
- Install all dependencies from requirements.txt
- Check Python version (requires 3.8+)
- Ensure port 5000 is not occupied by another application

## Technical Support

If you encounter problems:
1. Check the application console output
2. Ensure all dependencies are installed correctly
3. Check that the model format meets requirements

## License

This project is distributed under the terms of the MIT License.
You are free to use, modify, and distribute the software in accordance with the license conditions.
For full details, see the [LICENSE](LICENSE) file included in the repository.
