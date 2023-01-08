import re
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
def tag_to_role_of_speech(tag):
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None
def tags_to_roles_of_speech(tags):
    roles=[]
    for tag in tags:
        roles.append(tag_to_role_of_speech(tag[1]))
    return roles
def skewed_sen_sim(sentence1,tags1,sentence2,tags2):
    l1=len(sentence1)
    l2=len(sentence2)
    sum=0
    for i in range(l1):
        best_match=0
        for j in range(l2):
            sim=word_sim(sentence1[i],tags1[i],sentence2[j],tags2[j])
            if best_match<sim:
                best_match=sim
        sum+=best_match
    sum_2_to_1=0
    return sum/l1
def basic_sen_sim(sentence1,tags1,sentence2,tags2):
    return (skewed_sen_sim(sentence1,tags1,sentence2,tags2) + skewed_sen_sim(sentence2,tags2,sentence1,tags1)) / 2
def skewed_text_sim(list_of_sentences1, roles_of_sentences1, list_of_sentences2, roles_of_sentences2):
    l1 = len(list_of_sentences1) -1
    l2 = len(list_of_sentences2) -1
    sum = 0
    for i in range(l1):
        best_match = 0
        for j in range(l2):
            sim = basic_sen_sim(list_of_sentences1[i],roles_of_sentences1[i], list_of_sentences2[j], roles_of_sentences2[j])
            if best_match < sim:
                best_match = sim
        sum += best_match
    sum_2_to_1 = 0
    return sum / l1
def basic_text_sim(list_of_sentences1, roles_of_sentences1, list_of_sentences2, roles_of_sentences2):
    sim=0
    l1 = len(list_of_words1) - 1  # last split is empty
    l2 = len(list_of_words2) - 1
    for i in range(l1):
        for j in range(l2):
          return (skewed_text_sim(list_of_words1, roles_of_words1, list_of_words2, roles_of_words2) + skewed_text_sim(list_of_words2, roles_of_words2, list_of_words1, roles_of_words1)) / 2
def word_sim(word1,part1,word2,part2):
    synset1=None
    synset2=None
    if part1 == None:
        if wn.synsets(word1):
            synset1 = wn.synsets(word1)[0]
    else:
        if wn.synsets(word1,part1):
            synset1 = wn.synsets(word1,part1)[0]
    if part2 == None:
        if wn.synsets(word2):
            synset2 = wn.synsets(word2)[0]
    else:
        if wn.synsets(word2, part2):
            synset2 = wn.synsets(word2, part2)[0]
    if synset1 and synset2:
        return wn.wup_similarity(synset1,synset2)
    return 0

fd1=open("text1.txt")
fd2=open("text2.txt")
text1=fd1.read()
text2=fd2.read()
list_of_sentences1 = re.split("[.;]", text1)
list_of_sentences2 = re.split("[.;]", text2)
list_of_words1 = [word_tokenize(sentence) for sentence in
                  list_of_sentences1]  # actually list of list of words, first list is the first sentence,
list_of_words2 = [word_tokenize(sentence) for sentence in
                  list_of_sentences2]  # first element in the inner list is the first word of the sencence
print(pos_tag(list_of_words1[0]))
roles_of_words1 = [tags_to_roles_of_speech(pos_tag(sentence)) for sentence in list_of_words1]
roles_of_words2 = [tags_to_roles_of_speech(pos_tag(sentence)) for sentence in list_of_words2]
print(roles_of_words1)
print(basic_text_sim(list_of_words1, roles_of_words1, list_of_words2, roles_of_words2))



