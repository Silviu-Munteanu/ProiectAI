import re
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag

from Tagger import Tagger


class SimilarityAnalyser:
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    @staticmethod
    def __create_list_of_sentences_from_text(text: str):
        return re.split("[.;]", text)

    @staticmethod
    def __get_word_similarity(word1, part1, word2, part2):
        synset1 = wn.synsets(word1, part1)
        synset2 = wn.synsets(word2, part2)
        if synset1 and synset2:
            return wn.wup_similarity(synset1[0], synset2[0])
        return 0

    @staticmethod
    def __get_skewed_sentence_similarity(sentence1, tags1, sentence2, tags2):
        l1 = len(sentence1)
        l2 = len(sentence2)
        sum = 0
        for i in range(l1):
            best_match = 0
            for j in range(l2):
                sim = SimilarityAnalyser.__get_word_similarity(sentence1[i], tags1[i], sentence2[j], tags2[j])
                if best_match < sim:
                    best_match = sim
            sum += best_match
        return sum / l1

    @staticmethod
    def __get_basic_sentence_similarity(sentence1, tags1, sentence2, tags2):
        return (SimilarityAnalyser.__get_skewed_sentence_similarity(sentence1, tags1, sentence2, tags2)
                + SimilarityAnalyser.__get_skewed_sentence_similarity(sentence2, tags2, sentence1,
                                                                      tags1)) / 2  # the arithmetic mean

    @staticmethod
    def __get_skewed_text_similarity(list_of_sentences1, roles_of_sentences1, list_of_sentences2, roles_of_sentences2):
        l1 = len(list_of_sentences1)
        l2 = len(list_of_sentences2)
        sum = 0
        for i in range(l1):
            best_match = 0
            for j in range(l2):
                sentence_similarity = SimilarityAnalyser.__get_basic_sentence_similarity(list_of_sentences1[i],
                                                                                         roles_of_sentences1[i],
                                                                                         list_of_sentences2[j],
                                                                                         roles_of_sentences2[j])
                if best_match < sentence_similarity:
                    best_match = sentence_similarity
            sum += best_match
        return sum / l1  # the arithmetic mean of the similarities

    @staticmethod
    def __get_words_and_their_roles_from_text(text: str):
        list_of_sentences1 = SimilarityAnalyser.__create_list_of_sentences_from_text(text)
        list_of_list_of_words1 = [word_tokenize(sentence) for sentence in
                                  list_of_sentences1]  # actually list of list of words, first list is the first sentence
        list_of_list_of_words1.pop()  # popping the last word because it's always empty
        list_of_list_of_roles_of_words1 = [Tagger.tags_to_roles_of_speech(pos_tag(sentence)) for sentence in
                                           list_of_list_of_words1]
        return list_of_list_of_words1, list_of_list_of_roles_of_words1

    def get_basic_text_similarity(self):
        list_of_list_of_words1, list_of_list_of_roles_of_words1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text1)
        list_of_list_of_words2, list_of_list_of_roles_of_words2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text2)
        print(list_of_list_of_words1)  # FOR DEBUG PURPOSE
        print(list_of_list_of_roles_of_words1) # FOR DEBUG PURPOSE
        text1_text2_similarity = SimilarityAnalyser.__get_skewed_text_similarity(list_of_list_of_words1,
                                                                                 list_of_list_of_roles_of_words1,
                                                                                 list_of_list_of_words2,
                                                                                 list_of_list_of_roles_of_words2)
        text2_text1_similarity = SimilarityAnalyser.__get_skewed_text_similarity(list_of_list_of_words2,
                                                                                 list_of_list_of_roles_of_words2,
                                                                                 list_of_list_of_words1,
                                                                                 list_of_list_of_roles_of_words1)
        return (text1_text2_similarity + text2_text1_similarity) / 2  # the arithmetic mean of the similarities
