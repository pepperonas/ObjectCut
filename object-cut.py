import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QMimeData
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QMessageBox, QComboBox)
from rembg import remove


class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.is_selecting = False
        self.has_selection = False

        self.setStyleSheet("background-color: #1E1F2A; border: 1px solid #4A4B58;")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)

        # Aktiviere Drag & Drop
        self.setAcceptDrops(True)

    def setImage(self, image):
        self.original_image = image
        self.displayed_image = image.copy()
        self.updateDisplay()
        self.has_selection = False

    def updateDisplay(self):
        if hasattr(self, 'displayed_image'):
            h, w, c = self.displayed_image.shape
            q_img = QImage(self.displayed_image.data, w, h, w * c, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(q_img))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'original_image'):
            self.selection_start = event.pos()
            self.selection_end = event.pos()
            self.is_selecting = True
            self.has_selection = False

    def mouseMoveEvent(self, event):
        if self.is_selecting and hasattr(self, 'original_image'):
            self.selection_end = event.pos()
            self.displayed_image = self.original_image.copy()
            self.drawSelection()
            self.updateDisplay()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_selecting and hasattr(self, 'original_image'):
            self.selection_end = event.pos()
            self.is_selecting = False
            self.has_selection = True
            self.drawSelection()
            self.updateDisplay()

    def drawSelection(self):
        if hasattr(self, 'displayed_image'):
            h, w, _ = self.displayed_image.shape
            label_w, label_h = self.width(), self.height()

            # Convert UI coordinates to image coordinates
            img_x_scale = w / label_w
            img_y_scale = h / label_h

            x1 = max(0, min(int(self.selection_start.x() * img_x_scale), w - 1))
            y1 = max(0, min(int(self.selection_start.y() * img_y_scale), h - 1))
            x2 = max(0, min(int(self.selection_end.x() * img_x_scale), w - 1))
            y2 = max(0, min(int(self.selection_end.y() * img_y_scale), h - 1))

            # Sort coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Draw rectangle
            cv2.rectangle(self.displayed_image, (x1, y1), (x2, y2), (86, 180, 233), 2)

    def getSelectionCoordinates(self):
        if not self.has_selection:
            return None

        h, w, _ = self.original_image.shape
        label_w, label_h = self.width(), self.height()

        # Convert UI coordinates to image coordinates
        img_x_scale = w / label_w
        img_y_scale = h / label_h

        x1 = max(0, min(int(self.selection_start.x() * img_x_scale), w - 1))
        y1 = max(0, min(int(self.selection_start.y() * img_y_scale), h - 1))
        x2 = max(0, min(int(self.selection_end.x() * img_x_scale), w - 1))
        y2 = max(0, min(int(self.selection_end.y() * img_y_scale), h - 1))

        # Sort coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        return (x1, y1, x2, y2)

    # Neue Drag & Drop Methoden
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0:
                filepath = urls[0].toLocalFile()
                if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Signal to the parent to load this image
                    if self.parent:
                        self.parent.loadImageFromPath(filepath)
                        event.acceptProposedAction()


class ObjectExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Extractor")
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: #2C2E3B;
                color: #FFFFFF;
            }}
            QPushButton {{
                background-color: #4A4B58;
                color: #FFFFFF;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #5A5B68;
            }}
            QPushButton:pressed {{
                background-color: #3A3B48;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #4A4B58;
                height: 8px;
                background: #1E1F2A;
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: #5A5B68;
                border: 1px solid #5A5B68;
                width: 18px;
                margin: -2px 0;
                border-radius: 4px;
            }}
            QComboBox {{
                background-color: #4A4B58;
                color: #FFFFFF;
                border-radius: 4px;
                padding: 8px 16px;
            }}
            QLabel {{
                color: #FFFFFF;
            }}
        """)

        self.image_path = None
        self.original_image = None
        self.initUI()

        # Aktiviere Drag & Drop für die gesamte App
        self.setAcceptDrops(True)

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Top control bar
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.loadImage)

        self.extraction_method = QComboBox()
        self.extraction_method.addItems(["AI Background Removal", "Rectangle Selection"])

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.extraction_method)
        control_layout.addStretch()

        # Image display
        self.canvas = ImageCanvas(self)  # Pass self as parent

        # Drop zone Hinweis
        self.drop_hint = QLabel("Drag & Drop Image Here")
        self.drop_hint.setAlignment(Qt.AlignCenter)
        self.drop_hint.setStyleSheet("color: #5A5B68; font-size: 18px; font-weight: bold;")

        # Bottom control bar
        bottom_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Object")
        self.extract_btn.clicked.connect(self.extractObject)
        self.extract_btn.setEnabled(False)

        self.save_btn = QPushButton("Save PNG")
        self.save_btn.clicked.connect(self.savePNG)
        self.save_btn.setEnabled(False)

        bottom_layout.addWidget(self.extract_btn)
        bottom_layout.addWidget(self.save_btn)

        # Add all layouts to main layout
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.drop_hint)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addLayout(bottom_layout)

        self.setCentralWidget(main_widget)
        self.setMinimumSize(800, 600)

    def loadImage(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if filepath:
            self.loadImageFromPath(filepath)

    def loadImageFromPath(self, filepath):
        """Laden eines Bildes aus einem Dateipfad (für Drag & Drop und regulären Dialog)"""
        self.image_path = filepath
        self.original_image = cv2.imread(filepath)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        self.canvas.setImage(self.original_image)
        self.extract_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

        # Hinweis ausblenden nach dem Laden
        self.drop_hint.setVisible(False)

    def extractObject(self):
        if self.original_image is None:
            return

        method = self.extraction_method.currentText()

        if method == "Rectangle Selection":
            coords = self.canvas.getSelectionCoordinates()
            if coords:
                x1, y1, x2, y2 = coords
                # Create a mask for the selected region
                mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                # Apply the mask
                self.extracted_image = self.original_image.copy()
                background = np.ones_like(self.extracted_image) * 255  # White background
                alpha_channel = np.where(mask[:, :, np.newaxis] == 0, 0, 255).astype(np.uint8)

                # Convert to RGBA
                self.extracted_image = np.dstack((self.extracted_image, alpha_channel))

                # Display the extracted object
                rgb_display = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
                mask_display = cv2.merge([mask, mask, mask])
                display_img = cv2.bitwise_and(rgb_display, mask_display)
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

                self.canvas.setImage(display_img)
                self.save_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "Please select an area first")

        elif method == "AI Background Removal":
            # Use rembg for AI-based background removal
            try:
                # Convert to BGR for rembg
                input_img = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)

                # Process with rembg
                output = remove(input_img)

                # Convert back to RGB for display
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)

                # Save for later
                self.extracted_image = output_rgb

                # Display (RGB only)
                display_img = output_rgb[:, :, :3].copy()

                self.canvas.setImage(display_img)
                self.save_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error during AI extraction: {str(e)}")

    def savePNG(self):
        if not hasattr(self, 'extracted_image'):
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            "",
            "PNG Files (*.png)"
        )

        if filepath:
            if not filepath.endswith('.png'):
                filepath += '.png'

            # Save with alpha channel
            cv2.imwrite(filepath, cv2.cvtColor(self.extracted_image, cv2.COLOR_RGBA2BGRA))
            QMessageBox.information(self, "Success", f"Image saved to {filepath}")

    # Drag & Drop Methoden für das Hauptfenster
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) > 0:
                filepath = urls[0].toLocalFile()
                if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.loadImageFromPath(filepath)
                    event.acceptProposedAction()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectExtractorApp()
    window.show()
    sys.exit(app.exec_())