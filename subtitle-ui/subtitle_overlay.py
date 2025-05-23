import sys
import argparse
import os
import signal # For graceful shutdown on Ctrl+C
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QPoint, QTimer # Removed pyqtSignal, QObject, Added QTimer
from PyQt6.QtGui import QPalette, QColor, QFont


class SubtitleWindow(QWidget):
    def __init__(self, file_path=None, font_size=18, text_color="white",
                 font_family="Arial", bg_color="black", bg_opacity=0.5):
        super().__init__()
        self.file_path = file_path if file_path else "text.txt" # Default to text.txt
        self.font_size = font_size
        self.text_color = text_color
        self.font_family = font_family
        self.bg_color_name = bg_color
        self.bg_opacity = bg_opacity
        self.current_text = "Waiting for text..."

        self.init_ui()
        self.update_text_from_file() # Initial load

        # Timer to update text every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_text_from_file)
        self.timer.start(2000) # 2000 milliseconds = 2 seconds

    def init_ui(self):
        # Window Flags: Frameless and Always-on-Top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        
        # For semi-transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Main window is now fully transparent due to WA_TranslucentBackground.
        # The label will have its own background.

        # Label to display subtitle text
        self.label = QLabel(self.current_text, self) # Use current_text
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setAutoFillBackground(True) # Crucial for the label to draw its own background
        
        # Basic styling (can be expanded later)
        font = QFont(self.font_family, self.font_size)
        self.label.setFont(font)
        
        label_palette = self.label.palette()
        try:
            # Set text color
            text_qcolor = QColor(self.text_color)
            if not text_qcolor.isValid():
                print(f"Warning: Invalid text color '{self.text_color}'. Using white.")
                text_qcolor = QColor("white")
            label_palette.setColor(QPalette.ColorRole.WindowText, text_qcolor)

            # Set label background color (semi-transparent black)
            label_bg_qcolor = QColor(self.bg_color_name) # Should be "black" by default
            if not label_bg_qcolor.isValid():
                print(f"Warning: Invalid label background color '{self.bg_color_name}'. Using black.")
                label_bg_qcolor = QColor("black")
            
            label_alpha = int(max(0, min(1, self.bg_opacity)) * 255) # Default 0.5 -> 128
            label_bg_qcolor.setAlpha(label_alpha)
            label_palette.setColor(QPalette.ColorRole.Window, label_bg_qcolor) # Window role for background

        except Exception as e:
            print(f"Warning: Could not set text color or label background. Using defaults. Error: {e}")
            # Fallback for text color
            label_palette.setColor(QPalette.ColorRole.WindowText, QColor("white"))
            # Fallback for label background
            label_palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0, 128)) # Semi-transparent black
        self.label.setPalette(label_palette)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Set initial size (can be adjusted based on text later)
        self.set_initial_size_and_position()
        
        self.setWindowTitle("Subtitle Overlay")
        self.show()

    def set_initial_size_and_position(self):
        # Get screen geometry
        screen_geometry = QApplication.primaryScreen().geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Initial size (adjust as needed, or make dynamic later)
        # For now, a fixed width and height that can accommodate some text
        window_width = 600
        window_height = 70 
        self.setGeometry(
            (screen_width - window_width) // 2,  # Center horizontally
            screen_height - window_height - 50, # 50px from bottom
            window_width,
            window_height
        )
        # Adjust label padding/margins if necessary for better appearance
        self.label.setContentsMargins(10, 5, 10, 5)

        # For dragging
        self._drag_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def read_file_content(self):
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Filter out empty or whitespace-only lines and strip them
                non_empty_lines = [line.strip() for line in lines if line.strip()]

                if not non_empty_lines:
                    return "Waiting for text..."

                # Get only the very last non-empty line
                last_line_from_file = non_empty_lines[-1]
                
                words = last_line_from_file.split()
                formatted_lines = []
                for i in range(0, len(words), 10):
                    chunk = words[i:i+10]
                    formatted_lines.append(" ".join(chunk))
                
                return "\n".join(formatted_lines)
            except Exception as e:
                print(f"Error reading or processing file {self.file_path}: {e}")
                return "Error processing file."
        elif not self.file_path:
             return "No file path specified."
        return "File not found or empty." # If file doesn't exist or is empty after strip

    def update_text_from_file(self):
        new_text = self.read_file_content()
        # Always update, even if text is the same, to handle potential file read errors clearing
        self.current_text = new_text
        self.label.setText(self.current_text)
        # Optional: Adjust window size based on new text content if desired
        # self.label.adjustSize() # Adjusts label size
        # self.adjustSize()       # Adjusts window size to fit content

    # Removed init_file_observer method
    # Removed closeEvent method (or remove observer parts if it had other logic)
    # If closeEvent only handled observer, it can be removed. Otherwise, edit it.
    # For simplicity, assuming it was only for observer and can be removed.
    # If other cleanup is needed in closeEvent, it should be preserved.


if __name__ == '__main__':
    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print('Ctrl+C pressed. Exiting...')
        QApplication.quit()

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Subtitle Overlay Application")
    parser.add_argument("file_path", nargs='?', default="text.txt", # Default to text.txt
                        help="Path to the text file to display (default: text.txt).")
    parser.add_argument("--font-size", type=int, default=18,
                        help="Font size for the subtitle text (default: 18).")
    parser.add_argument("--text-color", type=str, default="white",
                        help="Color for the subtitle text (e.g., 'white', 'red', '#FFFF00'; default: 'white').")
    parser.add_argument("--font-family", type=str, default="Arial",
                        help="Font family for the subtitle text (default: 'Arial').")
    parser.add_argument("--bg-color", type=str, default="black",
                        help="Background color (e.g., 'black', '#333333'; default: 'black').")
    parser.add_argument("--bg-opacity", type=float, default=0.5,
                        help="Background opacity (0.0 to 1.0; default: 0.5).")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    # Check if PyQt6 is available
    try:
        from PyQt6 import QtCore
    except ImportError:
        print("PyQt6 is not installed. Please install it with: pip install PyQt6")
        # No app.exec() if QApplication not created or if critical components missing
        sys.exit(1)
    
    # Removed Watchdog related checks as it's no longer used.

    window = SubtitleWindow(file_path=args.file_path,
                            font_size=args.font_size,
                            text_color=args.text_color,
                            font_family=args.font_family,
                            bg_color=args.bg_color,
                            bg_opacity=args.bg_opacity)
    sys.exit(app.exec())