# Object Extractor

A modern PyQt5-based tool for extracting objects from images with transparency.

## Features

- **Multiple Extraction Methods**:
    - AI-based background removal with [rembg](https://github.com/danielgatis/rembg)
    - Manual rectangle selection
- **User-Friendly Interface**:
    - Drag & drop support for images
    - Rectangle selection by mouse dragging
    - Dark theme for comfortable use
- **Output**:
    - Save as PNG with transparency

## Installation

### Prerequisites

- Python 3.6 or higher
- PyQt5
- OpenCV (cv2)
- numpy
- rembg (for AI-based background removal)

### Setup

1. Clone the repository or download the source code
2. Install the dependencies:

```bash
pip install -r requirements.txt 
```

3. Start the application:

```bash
python3 object_extractor.py
```

## Usage

1. **Load Image**:
    - Click on "Load Image" and select an image
    - OR drag and drop an image directly into the application

2. **Choose Extraction Method**:
    - **AI Background Removal**: Automatic removal of the image background using AI
    - **Rectangle Selection**: Rectangular selection of the area to be extracted

3. **Extract Object**:
    - For Rectangle Selection: Drag a rectangle around the desired object with your mouse
    - Click on "Extract Object"

4. **Save Result**:
    - Click on "Save PNG" to save the extracted object with transparency

## Technical Details

- **Framework**: PyQt5 for the user interface
- **Image Processing**: OpenCV and numpy
- **AI Background Removal**: rembg library
- **Transparency**: Alpha channel processing for PNG output

## Image Format Requirements

The tool supports the following input formats:

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)

The output format is always PNG with transparency (RGBA).

## Tips

- AI Background Removal works best with clearly identifiable foreground objects.
- For complex cutouts with very specific areas, use the Rectangle Selection.
- Large images are automatically scaled for display but processed and saved at their original resolution.

## Troubleshooting

- **AI Background Removal doesn't work**: Make sure rembg is correctly installed and functioning.
- **UI display issues**: Some high-DPI displays may have display problems. Adjust your system scaling settings.