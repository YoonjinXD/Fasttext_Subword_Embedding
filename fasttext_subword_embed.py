import torch
import numpy as np
import random
from random import shuffle
from collections import Counter
import argparse

# Attributes
# sub_dict OR sub_indices [word_index] = (word, [subwords])
# embedding = hash table

def NGram_Hashing(str):

    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    max_bucket = 2100000
    for s in str:
        hval = hval ^ ord(s)
        hval = (hval * fnv_32_prime) % max_bucket

    return hval

def Ngram(ngram, word):
    set_of_ngram = []
    for index, character in enumerate(word):
        if index == 0:
            set_of_ngram.append(word[0:ngram])
        elif (index + ngram) > (len(word)-1):
            set_of_ngram.append(word[index:])
            break
        else:
            set_of_ngram.append(word[index:(index + ngram)])

    return set_of_ngram

def NGram_Indicing(w2i, vocab, ngram):

    subword_indices = {}
    BOW, EOW = ('<', '>')

    # Testing mode. For OOV words without corresponding index.
    if w2i == 0:
        for i, word in enumerate(vocab):
            subword = BOW + word + EOW
            subword_indices[i] = (word, Ngram(ngram, subword))

    # Normal mode. For a training set with index.
    else:
        for word in vocab:
            subword = BOW + word + EOW
            subword_indices[w2i[word]] = (word, Ngram(ngram, subword))

    return subword_indices


def get_subword_idx(sub_dict, word):
    idx_set =[]
    set_ngrams = None

    #Testing mode. The input 'word' is already a subword.
    if sub_dict == 0 :
        set_ngrams = word
    else:
        set_ngrams = (sub_dict[word])[1]

    for subword in set_ngrams:
        idx_set.append(NGram_Hashing(subword))

    # Output: index of subwords. size= torch.tensor((N, )) when N = num of ngrams
    return torch.tensor(idx_set).view(-1,)


def Most_Similar(input, sub_dict, embedding, ngram):

    input_sub_dict = NGram_Indicing(0, input, ngram)

    for i, (input_word, input_subword) in input_sub_dict.items():
        candidate = [0, 0, 0, 0, 0]
        similar_list = [0, 0, 0, 0, 0]

        pred_vec = embedding[get_subword_idx(0, input_subword)].sum(0)

        # Search 5 most similar words. *** Without sorting! ***
        for word_idx, (word, subwords) in sub_dict.items():

            if input_word == word:
                continue

            word_vec = embedding[get_subword_idx(sub_dict, word_idx)].sum(0)
            cosine_sim = torch.cosine_similarity(word_vec, pred_vec, 0)

            if (cosine_sim > min(candidate)):
                youareout_idx = candidate.index(min(candidate))
                candidate[youareout_idx] = cosine_sim
                similar_list[youareout_idx] = word

        print(i, input_word, " -> ", similar_list)


def subsampling(word_seq):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################

    words_count = Counter(word_seq)
    total_count = len(word_seq)
    words_freq = {word: count/total_count for word, count in words_count.items()}

    prob = {}

    for word in words_freq:
        prob[word] = 1 - np.sqrt(0.00001/words_freq[word])

    subsampled = [word for word in word_seq if random.random() < (1 - prob[word])]
    return subsampled


def SG(subwords, inputMatrix, outputMatrix):
# centerWord : Subword Index of a centerword (type: torch.tensor(P,))
# inputMatrix : Weight matrix of input = Hash table (type:torch.tesnor(K, D))
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(N,D))

    _, D = inputMatrix.size()
    inputVector = inputMatrix[subwords].sum(0)
    out = outputMatrix.mm(inputVector.view(D, 1))

    out_for_loss = -out
    out_for_loss[0] = -out_for_loss[0]

    loss = -torch.log(torch.sigmoid(out_for_loss)).sum()

    grad = torch.sigmoid(out)
    grad[0] -= 1

    grad_in = grad.view(1,-1).mm(outputMatrix)
    grad_out = grad.mm(inputVector.view(1,-1))

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, sub_dict, stats, NS=20, dimension=10000, learning_rate=0.025, epoch=3):

    max_bucket = 2100000
    W_in = torch.randn(max_bucket, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples", len(input_seq))
    stats = torch.LongTensor(stats)

    for _ in range(epoch):

        for inputs, output in zip(input_seq,target_seq):
            i+=1

            inputs = get_subword_idx(sub_dict, inputs)
            random_idx = torch.randint(0, len(stats), size=(NS,))
            neg_sample = (stats.view(-1, ))[random_idx]
            activated = torch.cat([torch.tensor([output]), neg_sample], 0)

            L, G_in, G_out = SG(inputs, W_in, W_out[activated])
            W_in[inputs] -= learning_rate * G_in.squeeze()
            W_out[activated] -= learning_rate * G_out

            losses.append(L.item())
            if i%100000==0:
                avg_loss=sum(losses)/len(losses)
                print("Iteration:", i, "Loss : %f" %(avg_loss,))
                losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('ngram', metavar='n-grams', type=int,
                        help='n-gram number')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')

    args = parser.parse_args()
    part = args.part
    ns = args.ns
    ngram = args.ngram

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    words = subsampling(words)

    #Make training set
    print("build training set...")
    input_set = []
    target_set =[]
    window_size = 5
    for j in range(len(words)):
        if j < window_size:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [0 for _ in range(window_size - j)] + [w2i[words[k]] for k in range(j)] + [
                w2i[words[j + k + 1]] for k in range(window_size)]
        elif j >= len(words) - window_size:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[len(words) - k - 1]] for k in
                                                                                 range(len(words) - j - 1)] + [0 for _
                                                                                                               in range(
                    j + window_size - len(words) + 1)]
        else:
            input_set += [w2i[words[j]] for _ in range(window_size * 2)]
            target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[j + k + 1]] for k in
                                                                                 range(window_size)]

    print("Vocabulary size")
    print(len(w2i))

    # Processing Subword
    sub_dict = NGram_Indicing(w2i, vocab, ngram)
    print("Complete creating a subword dictionary")

    # Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), sub_dict, freqtable, NS=ns, dimension=64, epoch=1, learning_rate=0.01)

    # Testint section
    test_input = ['narrow-mindedness', 'department', 'campfires', 'knowing', 'urbanize', 'imperfection', 'principality', 'abnormal', 'secondary', 'ungraceful']
    Most_Similar(test_input, sub_dict, emb, ngram)

main()
