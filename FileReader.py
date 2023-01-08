class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file_content(self):
        try:
            with open(self.file_path, "r+") as file:
                return file.read()
        except IOError:
            print(f"Could not open file {file.name}")
