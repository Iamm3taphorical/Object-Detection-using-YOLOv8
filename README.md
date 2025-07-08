# Object Detection using YOLOv8

A comprehensive Python pipeline using Ultralytics YOLOv8 for object detection in images. This system processes images, draws colored bounding boxes with class-confidence labels, and optional center checkmarks. Features include batch processing, configurable thresholds, customizable output directories, and Matplotlib visualization grids.

## 🚀 Features

- **Multi-Image Processing**: Batch process multiple images at once
- **Visual Annotations**: Colored bounding boxes, class labels, and confidence scores
- **Center Markers**: Optional center checkmarks on detected objects
- **Configurable Thresholds**: Adjustable confidence levels for detection
- **Grid Visualization**: Display results in organized Matplotlib grids
- **Output Management**: Organized saving to specified directories
- **Multiple Formats**: Support for various image formats (JPG, PNG, etc.)

## 📋 Requirements

Create a `requirements.txt` file with:

```
ultralytics>=8.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
numpy>=1.19.0
torch>=1.7.0
torchvision>=0.8.0
Pillow>=8.0.0
scipy>=1.5.0
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Iamm3taphorical/Object-Detection-using-YOLOv8.git
   cd Object-Detection-using-YOLOv8
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model (automatic on first run):**
   - The script will automatically download `yolov8n.pt` on first execution
   - Or manually download from [Ultralytics](https://github.com/ultralytics/ultralytics)

## 🎯 Usage

### Running the Python Script

```bash
python Object_Detection.py
```

### Running the Jupyter Notebook

```bash
jupyter notebook Object_Detection.ipynb
```

### Basic Usage Example

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model
model = YOLO('yolov8n.pt')

# Process images
image_paths = ['image1.jpg', 'image2.png', 'image3.jpeg']
results = model(image_paths)

# Display results
for result in results:
    result.show()  # Display with annotations
```

## 📁 Project Structure

```
Object-Detection-using-YOLOv8/
├── Object_Detection.py          # Main Python script
├── Object_Detection.ipynb       # Jupyter notebook version
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── images/                      # Input images folder
├── output/                      # Processed images output
└── models/                      # YOLO model files
```

## 🎛️ Configuration Options

### Confidence Threshold
- **Default**: 0.25
- **Range**: 0.1 - 1.0
- **Low (0.1-0.3)**: More detections, potential false positives
- **High (0.5-0.8)**: Fewer false positives, may miss objects

### Output Settings
- **Output Directory**: Configurable save location
- **Image Size**: Adjustable for display and processing
- **Grid Layout**: Customizable subplot arrangements

## 🎨 Features Detail

### Bounding Boxes
- Color-coded by object class
- Thickness and style customizable
- Class labels with confidence scores

### Center Checkmarks
- Green checkmarks at object centers
- Optional feature (can be disabled)
- Customizable size and color

### Visualization Grid
- Matplotlib-based display
- Multiple images in organized layout
- Before/after comparison support

## 🔧 Supported Object Classes

Detects 80 COCO dataset classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, horse, cow, elephant
- **Objects**: chair, table, laptop, phone, book
- **Food**: apple, banana, pizza, cake

## 📊 Performance Tips

### For Better Speed:
- Use YOLOv8n (nano) model
- Increase confidence threshold
- Process smaller image batches
- Use GPU acceleration if available

### For Better Accuracy:
- Use YOLOv8l or YOLOv8x models
- Lower confidence threshold
- Ensure high-quality input images
- Good lighting conditions

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'ultralytics'"**
   ```bash
   pip install ultralytics
   ```

2. **"CUDA out of memory"**
   - Use CPU inference: `model = YOLO('yolov8n.pt', device='cpu')`
   - Process smaller image batches

3. **"Image not found"**
   - Verify image paths are correct
   - Check supported formats: .jpg, .png, .jpeg, .bmp

4. **Poor detection results**
   - Adjust confidence threshold
   - Try different YOLOv8 model sizes
   - Ensure good image quality

## 📝 Example Output

The system generates:
- Annotated images with bounding boxes
- Class labels and confidence scores
- Optional center checkmarks
- Organized grid visualizations
- Console output with detection statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses YOLOv8 from Ultralytics (AGPL-3.0 License).

## 🆘 Support

For issues or questions:
- Check the troubleshooting section
- Verify all dependencies are installed
- Ensure input images are in supported formats
- Review the [Ultralytics documentation](https://docs.ultralytics.com/)

**Author**: Mahir Dyan  
**GitHub**: [@Iamm3taphorical](https://github.com/Iamm3taphorical)  
**Email**: mahirdyan30@gmail.com
