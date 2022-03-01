from nltk.tokenize import word_tokenize 
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

class LengthPenalty:
    def __init__(self, target_length):
        self.target_length = float(target_length)

    def score(self, utterance_1, utterance_2, extra=None):
        # scores = 1.0-length/self.target_length, penalizes very long responses but at the same time pushes for responses that are
        # as close to the target length as possible
        nlp = English()
        tokenizer = nlp.tokenizer
        tokenized_response = tokenizer(utterance_2)
        response_length = len(tokenized_response)
        # print(response_length)
        scores = 1.0 if response_length > self.target_length else 1.0-response_length/self.target_length
        # print(scores)
        return scores

example = LengthPenalty(20)
example_utterance_1 = 'I like playing tennis'
example_utterance_2 = 'What do I like to do?'
example.score(example_utterance_1, example_utterance_2)