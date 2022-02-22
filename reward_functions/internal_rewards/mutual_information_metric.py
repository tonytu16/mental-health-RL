import string
from nltk import ngrams
from expanding_contractions import contractions_dict
from expanding_contractions import expand_contractions
from nltk.translate.bleu_score import sentence_bleu
from itertools import chain
        
class PINCscore:
    def __init__(self, max_n_gram):
        self.max_n_gram = max_n_gram

    def ngram(self, document, max_n_gram):
        ngrams_list = []
        for i in range(1, max_n_gram + 1):
            splitted = ngrams(document.split(), i)
            ngrams_list.append(set(splitted))
        return ngrams_list

    def preprocess(self, text):
        #helper funfction for preprocessing text
        pre_processed_text = []
        for i in range(len(text)):
            expanded_text = (expand_contractions(text[i], contractions_dict)).lower()
            no_punc_text = expanded_text.translate(str.maketrans("", "", string.punctuation))  
            pre_processed_text.append(no_punc_text)
        return pre_processed_text

    def score(self, utterance_1, utterance_2, answers, lengths=None, extra=None):
        """
        The score function returns the PINC score for two documents.
    	With a maximum_lengths constraint, the function tokenizes the two
    	document and measure the level of similarity  between them.
    	The original implementation can be found here:
    	https://www.cs.utexas.edu/~ml/papers/chen.acl11.pdf
        """
        pre_processed_utterance_1 = self.preprocess(utterance_1)
        pre_processed_utterance_2 = self.preprocess(utterance_2)

        PINC_score_list = []
        for i in range(len(pre_processed_utterance_2)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_utterance_2[i].split()), len(pre_processed_utterance_1[i].split()), self.max_n_gram
            )

            # if Utterance is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            utterance_1_ngram_list = self.ngram(pre_processed_utterance_1[i], max_n_gram)
            utterance_2_ngram_list = self.ngram(pre_processed_utterance_2[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            for j in range(max_n_gram):
                overlap_count = 0
                for elem in utterance_2_ngram_list[j]:
                    if elem in utterance_1_ngram_list[j]:
                        overlap_count += 1
                PINC_score += 1 - overlap_count / len(utterance_2_ngram_list[j])
            PINC_score *= 1 / max_n_gram
            PINC_score_list.append(PINC_score)
        return PINC_score_list

    def score_two_utterances(self, utterance_ones, utterance_twos, lengths=None, extra=None):
        """
        The PINC scoring function specifically for two Utterance generation. 
        Instead of evaluating the level of similarity betweena context and the
        generated Utterances. This function instead evaluates the level of similarity
        between the two sets of generated functions
        """
        assert len(utterance_ones) == len(utterance_twos), "The number of Utterances must be equal"
        pre_processed_first_utterances = self.preprocess(utterance_ones)
        pre_processed_second_utterances = self.preprocess(utterance_twos)

        PINC_score_list = []
        for i in range(len(pre_processed_second_utterances)):
            # the N in the N-gram tokenization cannot exceed the number of words in the document
            max_n_gram = min(
                len(pre_processed_second_utterances[i].split()), len(pre_processed_first_utterances[i].split()), self.max_n_gram
            )

            # if utterance is blank, then score is 0
            if max_n_gram == 0:
                PINC_score_list.append(0)
                continue

            utterances_ones_ngram_list = self.ngram(pre_processed_first_utterances[i], max_n_gram)
            utterances_twos_ngram_list = self.ngram(pre_processed_second_utterances[i], max_n_gram)
            # we tokenize the groundtruth document and the prediction sentences
            # and create a 1-D array which contains all the n grams, where n ranges
            # from 1 to N
            PINC_score = 0
            #Utterance1 -> Utterance 1 PINC score
            PINC_score_reverse = 0
            #Utterance1 -> Utterance 2 PINC score
            for j in range(max_n_gram):
                overlap_count = 0
                overlap_count_reverse = 0
                for elem in utterances_twos_ngram_list[j]:
                    if elem in utterances_ones_ngram_list[j]:
                        overlap_count += 1
                for elem in utterances_ones_ngram_list[j]:
                    if elem in utterances_twos_ngram_list[j]:
                        overlap_count_reverse += 1
                PINC_score += 1 - overlap_count / len(utterances_twos_ngram_list[j])
                PINC_score_reverse += 1 - overlap_count_reverse / len(utterances_ones_ngram_list[j])
            PINC_score *= 1 / max_n_gram 
            PINC_score_reverse *= 1 / max_n_gram
            PINC_score_list.append((PINC_score+PINC_score_reverse)/2)
        print(PINC_score_list)
        return PINC_score_list

sent1 = ["I love playing tennis do you?"]
sent2 = ["I went to school today?"]
objects = PINCscore(5)
objects.score_two_utterances(sent1, sent2)