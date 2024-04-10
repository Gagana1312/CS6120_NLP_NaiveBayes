import pickle
#import text_classifier
import nbimporter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Load the model parameters from a local file
def load_model_parameters(file_path):
    with open(file_path, 'rb') as f:
        logprior, loglikelihood = pickle.load(f)
    return logprior, loglikelihood


def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 
    '''
    stemmer = PorterStemmer()
    english_stopwords = stopwords.words('english') ## Removing stopwords
    
    rhtml = re.sub(r'<.*?>', '', review)
    review = re.sub(r'http\S+|www\S+|https\S+', '', rhtml, flags=re.MULTILINE) ## link
    review = re.sub(r'[^\w\s]', '', review)
    review = re.sub(r'$[a-zA-Z]*', '', review) # words
    review_tokens = word_tokenize(review) ## tokenizing the review into words

    review_cleaned = []
    for word in review_tokens:
        stem_word = stemmer.stem(word)
        if stem_word.lower() not in review_cleaned:
            review_cleaned.append(stem_word.lower())

    # Convert the list of cleaned words to a single string
    #review_cleaned = ' '.join(review_cleaned)

    return review_cleaned

def find_occurrence(frequency, word, label):
    '''
    Params:
        frequency: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Return:
        n: the number of times the word with its corresponding label appears.
    '''
    n = frequency.get((word, label), 0)
  
    return n

def review_counter(output_occurrence, reviews, positive_or_negative):
    '''
    Params:
        output_occurrence: a dictionary that will be used to map each pair to its frequency
        reviews: a list of reviews
        positive_or_negative: a list corresponding to the sentiment of each review (either 0 or 1)
    Return:
        output: a dictionary mapping each pair to its frequency
    '''
    ## Steps :
    # define the key, which is the word and label tuple
    # if the key exists in the dictionary, increment the count
    # else, if the key is new, add it to the dictionary and set the count to 1
    
    for label, review in zip(positive_or_negative, reviews):
        split_review = clean_review(review)
        for word in split_review:
            kpair = (word, label)
            if kpair in output_occurrence:
                output_occurrence[kpair]+=1
            else:
                output_occurrence[kpair] = 1
   
    return output_occurrence

import math
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of reviews
        train_y: a list of labels correponding to the reviews (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0
    
    # calculate V, the number of unique words in the vocabulary
    vocab = set(key[0] for key in freqs.keys())
    V = len(vocab)

    # calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] == 1:
            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += freqs[pair]

        # else, the label is negative
        else:
            # increment the number of negative words by the count for this (word,label) pair
            num_neg += freqs[pair]

    # Calculate num_doc, the number of documents
    num_doc = len(train_x)

    # Calculate D_pos, the number of positive documents 
    pos_num_docs = (train_y==1).sum()

    # Calculate D_neg, the number of negative documents 
    neg_num_docs = (train_y==0).sum()

    # Calculate logprior
    logprior = np.log(pos_num_docs) - np.log(neg_num_docs)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = find_occurrence(freqs, word, 1)
        freq_neg = find_occurrence(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1)/(num_pos + V)
        p_w_neg = (freq_neg + 1)/(num_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)


    return logprior, loglikelihood

def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        message: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''
    
     # process the message to get a list of words
    word_l = clean_review(review)

    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob = total_prob + logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood.keys():
            # add the log likelihood of that word to the probability
            total_prob = total_prob + loglikelihood[word]

    if total_prob > 0:
        return 1
    else:
        return 0   


def main():
    # Load the model parameters
    model_file = 'naive_params.pkl'  
    logprior, loglikelihood = load_model_parameters(model_file)

    # Start the indefinite loop
    while True:
        # Get user input
        user_input = input("Enter a review ('X' to quit): ")

        # Check for exit condition
        if user_input.lower() == 'x':
            break

        # Preprocess the input
        cleaned_tokens = ' '.join(clean_review(user_input))
        

        # Predict the sentiment class
        sentiment = naive_bayes_predict(cleaned_tokens, logprior, loglikelihood)

        # Output the sentiment class
        print("Sentiment:", sentiment)
        print()

if __name__ == '__main__':
    main()