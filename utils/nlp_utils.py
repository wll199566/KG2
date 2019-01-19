"""
This script contains all the NLP helper functions.
"""
import re
import json
import codecs
import numpy as np

import nltk
import nltk.data
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def clean_data(string):
    """
     Argument:
             string: the string to be cleaned
    """
    string = re.sub(
        r"[^A-Za-z0-9,“”;?!=×−±:\.\(\)\-\*\/\'\<\>\[\]\—\+\{\}\"\’\–\%\'\`\|]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def nltk2wn_tag(nltk_tag):
    """
    Convert nltk tag to wordnet tag.
    This function is the helper for lemmatize_sentence.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    """
    Args:
        sentence: sentence string to be lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(res_words)


def split_into_sentences(string):
    """
    Split a string into sentences.
    Args:
        string: a string to be split.
    Output:
        a list containing each sentence.    
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #print ('\n'.join(tokenizer.tokenize(sentence)))
    return tokenizer.tokenize(string)


def load_glove(glove_path):
    """
    Argument: 
             glove_path: the path storing the downloaded glove.6B.50d.txt
    Output:
             a dictionary like {key: word, value: glove vector}         
    """
    # read in the whole glove and store as a dictionay
    # {key: word, value: glove vector}
    with open(glove_path, 'r') as fin:
        glove = {line.split()[0]: np.fromiter(map(float, line.split()[1:]), dtype=np.float)
                 for line in fin}

    return glove


def token_to_index(root_path):
    """
    Args:
        -root_path: the root folder containes the dict folders for hypo and supp,
                   like "../data".
    output:
        -the number of the tokens.
        -token_to_index dictionary {token: index}.      
    """
    hypo_dict_files = [root_path + "/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt",
                       root_path + "/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Dev-Hypothesis-dict.txt",
                       root_path + "/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Test-Hypothesis-dict.txt",
                       root_path + "/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Train-Hypothesis-dict.txt",
                       root_path + "/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-dict.txt",
                       root_path + "/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Test-Hypothesis-dict.txt"]

    supp_dict_files = [root_path + "/ARC-Supports-dict/ARC-Easy/ARC-Easy-Train-Supports-dict.txt",
                       root_path + "/ARC-Supports-dict/ARC-Easy/ARC-Easy-Dev-Supports-dict.txt",
                       root_path + "/ARC-Supports-dict/ARC-Easy/ARC-Easy-Test-Supports-dict.txt",
                       root_path + "/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Train-Supports-dict.txt",
                       root_path + "/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Dev-Supports-dict.txt",
                       root_path + "/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Test-Supports-dict.txt"]

    # construct a set to store unique tokens
    tokens = set()

    # to parse hypothesis dict files
    for file in hypo_dict_files:
        with open(file, 'r') as fin:
            for line in fin:
                ques_dict = json.loads(line)
                for dict_for_choice in ques_dict["triples_for_question"]:
                    for dict_for_sentence in dict_for_choice["triples_for_choice"]:
                        # Note in the dict file, there can be empty dict for sentence
                        if "sub" in dict_for_sentence.keys():
                            for token in word_tokenize(dict_for_sentence["sub"]):
                                tokens.add(token)
                        if "pred" in dict_for_sentence.keys():
                            for token in word_tokenize(dict_for_sentence["pred"]):
                                tokens.add(token)
                        if "time" in dict_for_sentence.keys():
                            for token in word_tokenize(dict_for_sentence["time"]):
                                tokens.add(token)
                        if "loc" in dict_for_sentence.keys():
                            for token in word_tokenize(dict_for_sentence["loc"]):
                                tokens.add(token)
                        if "obj" in dict_for_sentence.keys() and dict_for_sentence["obj"] != []:
                            for obj in dict_for_sentence["obj"]:
                                for token in word_tokenize(obj):
                                    tokens.add(token)

    # to parse support files
    for file in supp_dict_files:
        with open(file, "r") as fin:
            for line in fin:
                ques_dict = json.loads(line)
                for dict_for_choice in ques_dict["triples_for_question"]:
                    for dict_for_support in dict_for_choice["triples_for_choice"]:
                        for dict_for_sentence in dict_for_support["triples_for_support"]:
                            # Note in the dict file, there can be empty dict for sentence
                            if "sub" in dict_for_sentence.keys():
                                for token in word_tokenize(dict_for_sentence["sub"]):
                                    tokens.add(token)
                            if "pred" in dict_for_sentence.keys():
                                for token in word_tokenize(dict_for_sentence["pred"]):
                                    tokens.add(token)
                            if "time" in dict_for_sentence.keys():
                                for token in word_tokenize(dict_for_sentence["time"]):
                                    tokens.add(token)
                            if "loc" in dict_for_sentence.keys():
                                for token in word_tokenize(dict_for_sentence["loc"]):
                                    tokens.add(token)
                            if "obj" in dict_for_sentence.keys() and dict_for_sentence["obj"] != []:
                                for obj in dict_for_sentence["obj"]:
                                    for token in word_tokenize(obj):
                                        tokens.add(token)

    # construct the token2idx dictionary, here index from 1
    token2idx = {token: i for i, token in enumerate(tokens, 1)}
    # NOTE to add "Empty" to be 0 for
    # 1.Empty graph  
    # 2.packing the batch sentences of different sizes in LSTM
    token2idx["Empty"] = 0
    
    return len(tokens), token2idx


def construct_w2v_matrix(glove, embedding_dim, token_idx_dict, token_size):
    """
     Args:
         -glove: the loaded glove dictionary
         -embedding_dim: the embedding dim of glove vector
         -token_idx_dict: the dictionary of {token: index}
         -token_size: the number of tokens for this dataset 
     Output:
         -W: word matrix, each row is the representative vector for
             each word.         
    """
    W = np.zeros(shape=(token_size, embedding_dim), dtype='float32')
    for word, index in token_idx_dict.items():
        try:
            W[index, :] = glove[word]
        except KeyError:
            # if the word not in the glove, then use random vector whose
            # variant is the similar to the glove vectore.
            W[index, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

    return W


def load_token2idx(path):
    """
    Load token2idx dictionary.
    Args:
        -path: the path to the token2idx dictionary file, like "../data"
    Outputs:
        -token2idx_dict: token2idx dictionary.
    """    
    with open( path+"/Token2idx.json", "r" ) as fin:
        for line in fin:
            token2idx_dict = json.loads(line)
    print("Loaded word2idx dictionary!")        
    
    return token2idx_dict


def load_word_matrix(path):
    """
    Load word representation matrix.
    Args:
        -path: the path to the token2idx dictionary file, like "../data"
    Outputs:
        -word_mtx: pretrained word matrix.    
    """    
    with codecs.open(path+"/Wordmtx.json", 'r', encoding='utf-8') as fin:
       obj_text = fin.read()
       word_mtx = json.loads(obj_text)
       word_mtx = np.array(word_mtx)
       print("Loaded embedding matrix!")
    
    return word_mtx   



if __name__ == "__main__":

    # test clean_data
    # sentence = "my mom said {[(“I have a good son?!”)]}?!=×−±:|#$%^&*&("
    #sentence = "I am a good boy."
    #sentence_cleaned = clean_data(sentence)
    # print(sentence_cleaned)

    # test split_into_sentences
    #string = "Florida has a large supply of phosphate that ancient seas deposited millions of years ago. The phosphate contains the remains of animals that were deposited in layers on the sea floor. intrusive igneous type of rock did the phosphate become."
    #string = "I am a good boy."
    #sentence_list = split_into_sentences(string)
    # print(sentence_list)
    # for i, sentence in enumerate(sentence_list):
    #    sentence_list[i] = lemmatize_sentence(sentence)
    # print('\n'.join(sentence_list))

    # test token_to_index
    token_size, token2idx_dict = token_to_index("../data")
    print("There are", token_size, "tokens in the hypothesis and supports")
    with open("../data/Token2idx.json", 'w') as fout:
       json.dump(token2idx_dict, fout)
    #with open("../data/Token2idx.json", "r") as fin:
    #    for line in fin:
    #        token2idx_dict = json.loads(line)
    #print("good:", token2idx_dict["good"])
    #print("pencil:", token2idx_dict["pencil"])

    # test construct_w2v_matrix
    #with open("../data/Token2idx.json", "r") as fin:
    #   for line in fin:
    #       token2idx_dict = json.loads(line)
    token_size = len(token2idx_dict)
    glove = load_glove('../data/GloVe/glove.6B.50d.txt')
    W = construct_w2v_matrix(glove, 50, token2idx_dict, token_size)
    W_list = W.tolist()
    json.dump(W_list, codecs.open("../data/Wordmtx.json",'w', encoding='utf-8'))
    
    with codecs.open("../data/Wordmtx.json", 'r', encoding='utf-8') as fin:
        obj_text = fin.read()
        embedding_matrix = json.loads(obj_text)
        print("load embedding matrix")
        embedding_matrix = np.array(embedding_matrix)
    
    # to test whether the embedding_matrix has the same vector for each word as they are in glove.
    #glove = load_glove('../data/GloVe/glove.6B.50d.txt')
    print("embedding vector")
    print(glove["good"])
    print(glove["pencil"])
    print(embedding_matrix[token2idx_dict["good"]])
    print(embedding_matrix[token2idx_dict["pencil"]])