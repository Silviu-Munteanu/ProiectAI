import re
import numpy as np
import matplotlib.pyplot as plt

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
    def __get_sentence_from_index(text, index):
        return re.split("[.;]", text)[index]

    @staticmethod
    def __get_sentences_from_tuple(tupla, text1, text2):
        return re.split("[.;]", text1)[tupla[0]], re.split("[.;]", text2)[tupla[1]]

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
    def get_semantic_distance_matrix(list_of_sentences1, roles_of_sentences1, list_of_sentences2,
                                     roles_of_sentences2):
        output = []
        for i in range(len(list_of_sentences1)):
            temp = []
            for j in range(len(list_of_sentences2)):
                temp.append(
                    SimilarityAnalyser.__get_basic_sentence_similarity(list_of_sentences1[i], roles_of_sentences1[i],
                                                                       list_of_sentences2[j], roles_of_sentences2[j]))
            output.append(temp)
        return output

    @staticmethod
    def show_semantic_distance_matrix_heatmap(text1, text2):
        list_of_sentences1, roles_of_sentences1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(text1)
        list_of_sentences2, roles_of_sentences2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(text2)
        matrix = np.array(SimilarityAnalyser.get_semantic_distance_matrix(list_of_sentences1,
                                                                          roles_of_sentences1,
                                                                          list_of_sentences2,
                                                                          roles_of_sentences2))
        labels1 = [i for i in range(len(list_of_sentences1))]
        labels2 = [i for i in range(len(list_of_sentences2))]
        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        ax.set_xticks(np.arange(len(list_of_sentences1)), labels=labels1)
        ax.set_yticks(np.arange(len(list_of_sentences2)), labels=labels2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(list_of_sentences1)):
            for j in range(len(list_of_sentences2)):
                text = ax.text(j, i, round(matrix[i, j], 3), ha="center", va="center", color="w")
        ax.set_title("The smallest values for the semantic distance between sentences")
        fig.tight_layout()
        plt.show()

    def average_distance_all_cases(self):
        list_of_sentences1, roles_of_sentences1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text1)
        list_of_sentences2, roles_of_sentences2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text2)
        matrix = SimilarityAnalyser.get_semantic_distance_matrix(list_of_sentences1,
                                                                 roles_of_sentences1,
                                                                 list_of_sentences2,
                                                                 roles_of_sentences2)
        contor = 0
        medie = 0
        for i in matrix:
            for j in i:
                contor += 1
                medie += j
        return medie / contor

    def average_distance_after_greedy_assignation(self):
        list_of_sentences1, roles_of_sentences1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text1)
        list_of_sentences2, roles_of_sentences2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text2)
        temp = {}
        for i in range(len(list_of_sentences1)):
            for j in range(len(list_of_sentences2)):
                temp[(i, j)] = SimilarityAnalyser.__get_basic_sentence_similarity(list_of_sentences1[i],
                                                                                  roles_of_sentences1[i],
                                                                                  list_of_sentences2[j],
                                                                                  roles_of_sentences2[j])
        sorted_list = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True))
        matches = []
        contor = [i for i in range(
            len(list_of_sentences1) if len(list_of_sentences1) < len(list_of_sentences2) else len(list_of_sentences2))]
        switch = 0 if len(list_of_sentences1) < len(list_of_sentences2) else 1
        while sorted([t[switch] for t in matches]) != contor:
            for i in sorted_list.keys():
                mark = True
                for j in matches:
                    if i[0] == j[0] or i[1] == j[1]:
                        mark = False
                if mark:
                    matches.append(i)
        temp_sum = 0
        for i in matches:
            temp_sum += temp[i]
        return temp_sum / len(matches), matches, [temp[i] for i in matches]

    def closest_to_similarity_score(self, prag):
        list_of_sentences1, roles_of_sentences1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text1)
        list_of_sentences2, roles_of_sentences2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(self.text2)
        matrix = SimilarityAnalyser.get_semantic_distance_matrix(list_of_sentences1,
                                                                 roles_of_sentences1,
                                                                 list_of_sentences2,
                                                                 roles_of_sentences2)
        output = {}
        for i in matrix:
            for j in i:
                if j < prag:
                    output[(matrix.index(i), i.index(j))] = j, SimilarityAnalyser.__get_sentences_from_tuple((matrix.index(i), i.index(j)), self.text1, self.text2)
        return dict(sorted(output.items(), key=lambda item: item[1], reverse=True))

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
        list_of_list_of_words1, list_of_list_of_roles_of_words1 = SimilarityAnalyser.__get_words_and_their_roles_from_text(
            self.text1)
        list_of_list_of_words2, list_of_list_of_roles_of_words2 = SimilarityAnalyser.__get_words_and_their_roles_from_text(
            self.text2)
        print(list_of_list_of_words1)  # FOR DEBUG PURPOSE
        print(list_of_list_of_roles_of_words1)  # FOR DEBUG PURPOSE
        text1_text2_similarity = SimilarityAnalyser.__get_skewed_text_similarity(list_of_list_of_words1,
                                                                                 list_of_list_of_roles_of_words1,
                                                                                 list_of_list_of_words2,
                                                                                 list_of_list_of_roles_of_words2)
        text2_text1_similarity = SimilarityAnalyser.__get_skewed_text_similarity(list_of_list_of_words2,
                                                                                 list_of_list_of_roles_of_words2,
                                                                                 list_of_list_of_words1,
                                                                                 list_of_list_of_roles_of_words1)
        return (text1_text2_similarity + text2_text1_similarity) / 2  # the arithmetic mean of the similarities
