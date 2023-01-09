from FileReader import FileReader
from SimilarityAnalyser import SimilarityAnalyser

if __name__ == "__main__":
    file_reader1 = FileReader("Choose the first file to be analysed")
    file_reader2 = FileReader("Choose the second file to be analysed")
    similarity_analyzer = SimilarityAnalyser(file_reader1.read_file_content(), file_reader2.read_file_content())
    print(similarity_analyzer.average_distance_all_cases())
    print(similarity_analyzer.average_distance_after_greedy_assignation()[0])
    #print(similarity_analyzer.average_distance_after_greedy_assignation()[1])
    # print(similarity_analyzer.average_distance_after_greedy_assignation()[2])
    print(similarity_analyzer.closest_to_similarity_score(0.3))
    print(similarity_analyzer.get_text_similarity())
    similarity_analyzer.show_semantic_distance_matrix_heatmap(file_reader1.read_file_content(),
                                                              file_reader2.read_file_content())
