import os
import sys 
import sqlite3

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextBrowser, 
    QHBoxLayout, QTextEdit, QPushButton, QLabel, 
    QSizePolicy, QSpinBox, QMainWindow, QTabWidget,  
    )

# service imports 
from services.local_search import LocalSearchService

# remember the registry tip... implement later. 
if os.environ['LLMFLAG'] == 1:
    from services.chatbot import ChatServiceLLM
    chat_service = ChatServiceLLM()
else:
    from services.chatbot import ChatServiceBase
    chat_service = ChatServiceBase()

# global var -- convert intoa db service -- not thread safe
con = sqlite3.connect(os.environ["SEARCH_DB_PATH"]) # these are not thread safe

class SearchPage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()

        # there should be a way to set this with argparse. 
        self.search_service = LocalSearchService(con)

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

class ChatPage(QWidget):
    # sendData = pyqtSignal(dict) # define a signal 
    def __init__(self):
        super().__init__()
        self.chat_service = chat_service
       
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
        self.set_keynames.clicked.connect(self.callChat)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.KeyNameArea)
        bottom_bar.addWidget(self.set_keynames)

        # ---- MAIN LAYOUT ----
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.display)
        main_layout.addLayout(bottom_bar)

        self.setLayout(main_layout)

    def callChat(self):
        text = self.KeyNameArea.toPlainText()
        output = self.chat_service.chat(query = text)
        self.display.append(output)

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
        
        self.page2 = ChatPage() 
        tabs.addTab(self.page2, "Chat Page")
        
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
    con.close()

def main():
    # can the main app be a collection of classes?
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(cleanup)
    window = TabbedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()