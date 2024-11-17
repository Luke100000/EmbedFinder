import base64
import os
import sys

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QHBoxLayout,
    QPushButton,
)

from embedding import ImageEmbedder, AudioEmbedder
from files import FileManager, File, DEFAULT_IMAGE_PATTERNS, DEFAULT_SOUND_PATTERNS


class SearchWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Search Example")

        # Create layout
        layout = QVBoxLayout()

        first_row = QHBoxLayout()

        # Create input
        self.directory_bar = QLineEdit(self)
        self.directory_bar.setPlaceholderText("Root path...")
        # noinspection PyUnresolvedReferences
        first_row.addWidget(self.directory_bar)

        # Search button
        scan_button = QPushButton("Search")
        # noinspection PyUnresolvedReferences
        scan_button.clicked.connect(self.update_directory)
        first_row.addWidget(scan_button)

        # Create search bar
        second_row = QHBoxLayout()
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Type to search...")
        # noinspection PyUnresolvedReferences
        self.search_bar.textChanged.connect(self.update_list)
        second_row.addWidget(self.search_bar)

        # Create list widget to show results
        self.result_list = QListWidget(self)
        # noinspection PyUnresolvedReferences
        self.result_list.itemDoubleClicked.connect(self.open_file)

        # Embedding type buttons
        image_button = QPushButton("Image")
        # noinspection PyUnresolvedReferences
        image_button.clicked.connect(self.toggle_image_embedding)
        second_row.addWidget(image_button)

        audio_button = QPushButton("Audio")
        # noinspection PyUnresolvedReferences
        audio_button.clicked.connect(self.toggle_audio_embedding)
        second_row.addWidget(audio_button)

        layout.addLayout(first_row)
        layout.addLayout(second_row)
        layout.addWidget(self.result_list)

        self.setLayout(layout)

        self.manager = FileManager()

    def toggle_image_embedding(self):
        self.manager = FileManager(DEFAULT_IMAGE_PATTERNS, ImageEmbedder())

    def toggle_audio_embedding(self):
        self.manager = FileManager(DEFAULT_SOUND_PATTERNS, AudioEmbedder())

    def update_directory(self):
        root_path = self.directory_bar.text()
        self.manager.scan(root_path)
        self.update_list()

    def update_list(self):
        search_text = self.search_bar.text().lower()

        # Clear current list items
        self.result_list.clear()

        # Search through the files and add those that match the search text
        for file in self.manager.search(search_text):
            self.add_item_to_list(file)

    def add_item_to_list(self, file: File):
        # Create custom widget with image preview and metadata
        item_widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)

        # Image preview
        image = QImage()
        if file.thumbnail:
            image.loadFromData(base64.b64decode(file.thumbnail), "PNG")
            pixmap = QPixmap(image).scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setFixedSize(128, 128)
            layout.addWidget(image_label)

        # Metadata
        metadata_label = QLabel(os.path.basename(file.path))
        metadata_label.setStyleSheet("font-size: 10pt; color: gray;")
        layout.addWidget(metadata_label)

        # Set layout for the custom widget
        item_widget.setLayout(layout)

        # Set custom widget to list item
        list_item = QListWidgetItem(self.result_list)
        list_item.setData(Qt.UserRole, file.path)
        self.result_list.setItemWidget(list_item, item_widget)

        # Set minimum size for list item to give it more height (adjust as needed)
        list_item.setSizeHint(item_widget.sizeHint())

    def open_file(self, item):
        file_path = item.data(Qt.UserRole)
        if os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchWindow()
    window.show()
    sys.exit(app.exec_())
