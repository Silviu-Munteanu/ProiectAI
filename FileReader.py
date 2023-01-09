import os
from tkinter import filedialog
import tkinter as tk


class FileReader:
    def __init__(self, dialog_title: str):
        root = tk.Tk()
        root.withdraw()
        self.file_path = ""
        while self.file_path == "":
            self.file_path = filedialog.askopenfilename(title=dialog_title,
                                                        filetypes=[("TXT", "*.txt")], initialdir=os.getcwd())

    def read_file_content(self):
        try:
            with open(self.file_path, "r+") as file:
                return file.read()
        except IOError:
            print(f"Could not open file {file.name}")
