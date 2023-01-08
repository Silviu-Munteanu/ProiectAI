from FileReader import FileReader
from SimilarityAnalyser import SimilarityAnalyser

FILE1_NAME = "text1.txt"
FILE2_NAME = "text2.txt"


if __name__ == "__main__":
    file_reader1 = FileReader(FILE1_NAME)
    file_reader2 = FileReader(FILE2_NAME)
    similarity_analyser = SimilarityAnalyser(file_reader1.read_file_content(), file_reader2.read_file_content())
    print(f"The similarity between these two texts is:{similarity_analyser.get_basic_text_similarity()}")



