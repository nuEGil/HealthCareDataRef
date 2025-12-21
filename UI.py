import sys 
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextBrowser, 
    QHBoxLayout, QTextEdit, QPushButton, QLabel, 
    QSizePolicy, QSpinBox, QMainWindow, QTabWidget,  
    )

from services.local_search import LocalSearchService

class SearchPage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()

        # there should be a way to set this with argparse. 
        self.search_service = LocalSearchService()

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
        text = self.KeyNameArea.toPlainText()
        output = self.search_service.search(query = text)
        self.display.append(output)

class TabbedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tabbed PyQt6 App")
        self.setGeometry(200, 200, 700, 500)

        # Create Tab Widget
        tabs = QTabWidget()

        # Add tabs 1 and 2. 
        self.page1 = SearchPage() 
        tabs.addTab(self.page1, "Search Page")
        
        # # connect to sent signals 
        # self.page1.sendData.connect(self.page2.receiveData)

        # # Continue adding tabs. -- remember they just have to be widgets. 
        # second_page = QWidget()
        # second_layout = QVBoxLayout()
        # second_layout.addWidget(QLabel("This is another page!"))
        # second_page.setLayout(second_layout)
        # tabs.addTab(second_page, "Second Page")

        self.setCentralWidget(tabs)

def main():
    # can the main app be a collection of classes?
    app = QApplication(sys.argv)
    window = TabbedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()