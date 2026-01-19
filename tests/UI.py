import os
import sys 
import json
import sqlite3
import requests
import numpy as np
from PIL import Image
from PIL.ImageQt import fromqimage

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextBrowser, 
    QHBoxLayout, QTextEdit, QPushButton, QLabel, 
    QSizePolicy, QSpinBox, QMainWindow, QTabWidget,  
    QFileDialog, QGraphicsView, QGraphicsScene
    )

from PyQt6.QtGui import QImage, QPixmap,  QPainter
from PyQt6.QtCore import Qt, QRectF

from tests.utils.model_serve_tools import (mapmaker, make_padded_image, get_subimg_inds, 
                                           overlay_heatmap_simple)
'''Add a togle for the different cases on search page
search page should have 
1. code 
2. terms

this way you can post - 
code <ICD code prefix>
terms <user term list>'''

class SearchPage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()
        self.BASE_URL = "http://127.0.0.1:8000"
        # there should be a way to set this with argparse. 

        # ---- TOP: formatted display area ----
        self.display = QTextBrowser()
        self.display.setOpenExternalLinks(True)
        self.display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.display.setHtml("<b>Results will appear here</b>")

        # ---- BOTTOM: search bar ----
        self.KeyNameArea = QTextEdit()
        self.KeyNameArea.setPlaceholderText("Enter key words here")
        self.KeyNameArea.setFixedHeight(60)

        self.set_keynames = QPushButton("Send")
        self.set_keynames.clicked.connect(self.callSearch)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.KeyNameArea)
        bottom_bar.addWidget(self.set_keynames)

        # ---- MAIN LAYOUT ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display)
        main_layout.addLayout(bottom_bar)

        self.setLayout(main_layout)

    def callSearch(self):
        self.display.clear()
        text = self.KeyNameArea.toPlainText()
        endpoint = "/search"
        payload = {"text": text}  

        response = requests.post(self.BASE_URL + endpoint, json=payload)
        if response.status_code == 200:
            html_result = response.json().get("result")              
            self.display.append(html_result)

class KnowledgePage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()
        self.BASE_URL = "http://127.0.0.1:8001"
       
        # ---- TOP: formatted display area ----
        self.display = QTextBrowser()
        self.display.setOpenExternalLinks(True)
        self.display.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.display.setHtml("<b>Results will appear here</b>")

        # ---- BOTTOM: search bar ----
        self.KeyNameArea = QTextEdit()
        self.KeyNameArea.setPlaceholderText("Enter Chats Here")
        self.KeyNameArea.setFixedHeight(60)

        self.set_keynames = QPushButton("Send")
        self.set_keynames.clicked.connect(self.callknowledge)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.KeyNameArea)
        bottom_bar.addWidget(self.set_keynames)

        # ---- MAIN LAYOUT ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display)
        main_layout.addLayout(bottom_bar)

        self.setLayout(main_layout)

    def callknowledge(self):
        self.display.clear()
        text = self.KeyNameArea.toPlainText()
        endpoint = "/search"
        payload = {"text": text}  

        response = requests.post(self.BASE_URL + endpoint, json=payload)
        if response.status_code == 200:
            html_result = response.json().get("result")              
            self.display.append(html_result)

def numpy_to_pixmap(array):
    # 1. Ensure array is uint8 and in RGB format (not BGR)
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
        
    height, width, channels = array.shape
    bytes_per_line = channels * width
    
    # 2. Create QImage (specifying format is critical)
    q_img = QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    
    # 3. Convert to QPixmap
    return QPixmap.fromImage(q_img)

class ImagePage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()

        self.BASE_URL = "http://127.0.0.1:8002"
        self.path = None

        # ---- Graphics view ----
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.pixmap_item = None

        # ---- Top: load image ----
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        top_bar = QHBoxLayout()
        top_bar.addWidget(load_btn)
        top_bar.addStretch()

        # ---- Bottom: analyze ----
        analyze_btn = QPushButton("Analyze Image")
        analyze_btn.clicked.connect(self.callImageAnalyze)
        analyze_btn.setEnabled(False)
        self.analyze_btn = analyze_btn  # enable after load

        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        bottom_bar.addWidget(analyze_btn)

        # ---- Main layout ----
        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self.view, stretch=1)
        layout.addLayout(bottom_bar)
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.bmp)"
        )
        if not path:
            return
        
        self.path = path
        pixmap = QPixmap(path)
        
        if pixmap.isNull():
            # Handle corrupted/invalid image
            print(f"Failed to load image: {path}")
            return
        
        # Clear previous image
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        
        # Add new image
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)

        self.analyze_btn.setEnabled(True)

    def callImageAnalyze(self):
        self.analyze_btn.setEnabled(False) # turn off the button to avoid spamming
        endpoint = "/infer"
        payload = {"img_path": self.path}  
        print(payload)
        response = requests.post(self.BASE_URL + endpoint, json=payload)
        
        if response.status_code != 200:
            return
        
        # response is the json 
        html_result = response.json().get("points", [])          
        print(html_result[0])
        # # need some logic for no points. .. mybe. 
        # refactor later ---
        img0 = Image.open(self.path).convert("RGB")
        img0 = np.array(img0) / 255.0
    
        img_size = img0.shape[0]
        inds, pad_up = get_subimg_inds(img_size=img_size, stride=16)
        padimg0 = make_padded_image(img0, [img_size + pad_up, img_size + pad_up, 3])
        heatmap = mapmaker(padimg0, html_result, patch_size=128)
        overlay_ = overlay_heatmap_simple(img0, heatmap, alpha=0.5)
        over_pixmap = numpy_to_pixmap(np.array(overlay_))

        # --- update scene ---
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)

        self.pixmap_item = self.scene.addPixmap(over_pixmap)
        self.scene.setSceneRect(QRectF(over_pixmap.rect()))
        self.view.fitInView(self.scene.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)

        self.analyze_btn.setEnabled(True)
       
class TabbedApp(QMainWindow):
    def __init__(self,):
        super().__init__()
        self.setWindowTitle("Tabbed PyQt6 App")
        self.setGeometry(200, 200, 700, 500)

        # Create Tab Widget
        tabs = QTabWidget()

        # Add tabs 1 and 2. 
        self.page1 = SearchPage() 
        tabs.addTab(self.page1, "Search Page")
        
        self.page2 = KnowledgePage() 
        tabs.addTab(self.page2, "Knowledge Page")
        

        self.page3 = ImagePage() 
        tabs.addTab(self.page3, "Image Page")

        # # connect to sent signals 
        # self.page1.sendData.connect(self.page2.receiveData)

        # # Continue adding tabs. -- remember they just have to be widgets. 
        # second_page = QWidget()
        # second_layout = QVBoxLayout()
        # second_layout.addWidget(QLabel("This is another page!"))
        # second_page.setLayout(second_layout)
        # tabs.addTab(second_page, "Second Page")

        self.setCentralWidget(tabs)

def cleanup():
    pass

def main():
    # can the main app be a collection of classes?
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(cleanup)
    window = TabbedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()