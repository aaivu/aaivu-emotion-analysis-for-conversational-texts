#Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, RepeatVector, Embedding, LSTM, Concatenate, merge, Dropout, CuDNNLSTM, \
    Convolution1D, MaxPooling1D, Activation, GRU, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, Bidirectional, \
    Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import Input, concatenate
from keras.models import Model
from keras import optimizers
from keras.models import load_model
from nltk.tokenize import TweetTokenizer
import json, argparse, os
import re
import io
import sys
import fasttext
import emoji
import keras
from emoji import UNICODE_EMOJI
# from keras_self_attention import SeqSelfAttention
import string
# from nltk.corpus import stopwords
# import regex

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""
emojiDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer 
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
emoji2emoticons = {'😑': ':|', '😖': ':(', '😯': ':o', '😝': ':p', '😐': ':|',
                   '😈': ':)', '🙁': ':(', '😎': ':)', '😞': ':(', '♥️': '<3', '💕': 'love',
                   '😀': ':d', '😢': ":(", '👍': 'ok', '😇': ':)', '😜': ':p',
                   '💙': 'love', '☹️': ':(', '😘': ':)', '🤔': 'hmm', '😲': ':o',
                   '🙂': ':)', '\U0001f923': ':d', '😂': ':d', '👿': ':(', '😛': ':p',
                   '😉': ';)', '🤓': '8-)'}

def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    n = 0
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()

        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            # line.lower()
            allchars = [str for str in line]
            emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
            # clean_text = ' '.join([str for str in line.split() if not any(i in str for i in emoji_list)])
            repeatedChars = ['.', '?', '!', ',']
            repeatedChars = repeatedChars + emoji_list
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4])
            
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            indices.append(int(line[0]))
            conversations.append(conv.lower())
    
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations

def preprocessDataV(dataFilePath, mode, preprocess=False):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)


            if preprocess:
                stray_punct = ['‑', '-', "^", ":",
                    ";", "#", ")", "(", "*", "=", "\\", "/"]
                for punct in stray_punct:
                    conv = conv.replace(punct, "")

            if preprocess:
                processedData = regex.cleanText(conv.lower(), remEmojis=1).lower()
                processedData = processedData.replace("'", "")
                # Remove numbers
                processedData = ''.join([i for i in processedData if not i.isdigit()])
            else:
                processedData = conv.lower()

            indices.append(int(line[0]))
            conversations.append(processedData)

    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations



def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')

def cleanWord (word):
    final_list = []
    temp_word = ""
    for k in range(len(word)):
        if word[k] in UNICODE_EMOJI:
            # check if word[k] is an emoji
            if (word[k] in emoji2emoticons):
                final_list.append(emoji2emoticons[word[k]])
        else:
            temp_word = temp_word + word[k]

    # if len(temp_word) > 0:
    #     # "righttttttt" -> "right"
    #     temp_word = re.sub(r'(.)\1+', r'\1\1', temp_word)
    #     # correct spelling
    #     spell = SpellChecker()
    #     temp_word = spell.correction(temp_word)
    #     # lemmatize
    #     temp_word = WordNetLemmatizer().lemmatize(temp_word)
    #     final_list.append(temp_word)

    return final_list

def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'datastories.twitter.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    bad_words_glove_1 = set([])
    bad_words_glove_2 = set([])
    counter = 0

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            bad_words_glove_1.add(word)
            good_from_bad = cleanWord(word)
            # sum all word vectors, obtained after clean_word
            temp_embedding_vector = np.zeros((1, EMBEDDING_DIM))
            for www in good_from_bad:
                eee = embeddingsIndex.get(www)
                if eee is not None:
                    temp_embedding_vector = temp_embedding_vector + eee
            if not temp_embedding_vector.all():
                bad_words_glove_2.add(word)
            embeddingMatrix[i] = temp_embedding_vector
        if (counter % 1000 == 0):
            print(counter)
        counter += 1

        # print("Bad words in GloVe 1 - %d" % len(bad_words_glove_1))
        # print("Bad words in GloVe 2 - %d" % len(bad_words_glove_2))

    return embeddingMatrix

def getEmbeddingByBin(wordIndex):
    model = fasttext.load_model("250D_322M_tweets.bin")
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        v = model.get_word_vector(word)
        embeddingMatrix[i] = v
    return embeddingMatrix

def buildModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # embeddingLayerEmoji = Embedding(embeddingMatrixEmoji.shape[0],
    #                           EMBEDDING_DIM,
    #                           weights=[embeddingMatrix],
    #                           input_length=MAX_SEQUENCE_LENGTH,
    #                           trainable=False)
    #inp = Concatenate(axis=-1)([embeddingLayer,embeddingLayerEmoji])
    model = Sequential()
    # modelEmoji = Sequential()
    model.add(embeddingLayer)
    # modelEmoji.add(embeddingLayerEmoji)
    # modelFinal = Concatenate(axis=-1)([model, modelEmoji])
    #model.add(inp)
    # model.add(LSTM(LSTM_DIM, dropout=DROPOUT,return_sequences=True))
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model

def build3CnnModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    # model = Sequential()
    # model.add(embeddingLayer)
    # model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    # model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    #
    # rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=rmsprop,
    #               metrics=['acc'])
    ########################################################

    print('Training model.')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embeddingLayer(sequence_input)

    # add first conv filter
    embedded_sequences = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)

    # add second conv filter.
    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)

    # add third conv filter.
    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)

    # concate the conv layers
    alpha = concatenate([x, y, z])

    # flatted the pooled features.
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)

    # predictions
    preds = Dense(NUM_CLASSES, activation='softmax')(alpha)

    # build model
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta()

    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    return model

def build3Cnn_LstmModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    print('Training model.')

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embeddingLayer(sequence_input)

    # add first conv filter
    embedded_sequences = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)
    x = LSTM(50, dropout=0.2)(x)

    # add second conv filter.
    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)
    y = LSTM(50, dropout=0.2)(y)

    # add third conv filter.
    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)
    z = LSTM(50, dropout=0.2)(z)
    # concate the conv layers
    alpha = concatenate([x, y, z])

    # flatted the pooled features.
    alpha = Flatten()(alpha)

    # dropout
    alpha = Dropout(0.5)(alpha)

    # predictions
    preds = Dense(NUM_CLASSES, activation='softmax')(alpha)

    # build model
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta()

    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    return model

def build3LSTMModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=0.2, return_sequences=True))
    model.add(LSTM(LSTM_DIM, dropout=0.2, return_sequences=True))
    model.add(LSTM(LSTM_DIM, dropout=0.2))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.add(Activation('relu'))
    model.add(Activation('linear'))
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model

def build_CNNLSTM_Model(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic CNN-LSTM model
    """
    # Convolution parameters
    filter_length = 3
    nb_filter = 256
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    def swish(x):
        beta = 1.5  # 1, 1.5 or 2
        return beta * x * keras.backend.sigmoid(x)
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    # model.add(Dropout(0.2))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Bidirectional(LSTM(dropout=0.2,output_dim=output_size)))
    # model.add(Bidirectional(LSTM(dropout=0.2, output_dim=output_size, activation=rnn_activation,recurrent_activation=recurrent_activation)))
    # model.add(Flatten(Reshape((1, filters,))))
    # model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation ="sigmoid"))
    model.add(Dropout(0.25))
    sgd = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model
#working on
def build_CNNLSTM_Model_Concat(embeddingMatrix):
    # Convolution parameters
    filter_length = 3
    nb_filter = 256
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input1')

    e0 = Embedding(embeddingMatrix.shape[0],
                   EMBEDDING_DIM,
                   weights=[embeddingMatrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)
    emb1 = e0(x1)
    d=Dropout(0.2)
    o=d(emb1)
    c0 = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1)(o)
    c1 = Convolution1D(nb_filter=nb_filter,
                       filter_length=5,
                       border_mode=border_mode,
                       activation=cnn_activation,
                       subsample_length=1)(o)
    # c2 = Convolution1D(nb_filter=nb_filter,
    #                    filter_length=7,
    #                    border_mode=border_mode,
    #                    activation=cnn_activation,
    #                    subsample_length=1)(o)
    c=concatenate([c0,c1])
    p0=MaxPooling1D(pool_length=pool_length)(c)

    # p0 = MaxPooling1D(pool_length=pool_length)(p0)
    # p0=Reshape((1))(p0
    # emb1=Flatten()(emb1)
    # p0=Flatten()(p0)
    # print(p0.get_shape())
    # p0 = concatenate([emb1, p0])
    # p0=Reshape((1,))(p0)
    lstm = Bidirectional(LSTM(dropout=0.2,output_dim=output_size))
    out = lstm(p0)
    opt = Dense(NUM_CLASSES, activation='sigmoid')(out)
    dropout=Dropout(0.25)(opt)
    model = Model([x1], dropout)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def build3LSTMModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=0.2, return_sequences=True))
    model.add(LSTM(LSTM_DIM, dropout=0.2, return_sequences=True))
    model.add(LSTM(LSTM_DIM, dropout=0.2))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.add(Activation('relu'))
    model.add(Activation('linear'))
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model


def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(embedding_matrix.shape[0],EMBEDDING_DIM, weights=[embedding_matrix])(inp)
    x = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], EMBEDDING_DIM),
                      kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z=LSTM(LSTM_DIM,dropout=0.2)
    # z = Dropout(0.1)(z)

    outp = Dense(NUM_CLASSES, activation="sigmoid")(z)
    outp=Dropout(0.25)(outp)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_CNNLSTM_Model_With_Different_FilterSize(embeddingMatrix):
    # Convolution parameters
    filter_length = [1, 2, 3, 5]
    nb_filter = 256
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input1')

    e0 = Embedding(embeddingMatrix.shape[0],
                   EMBEDDING_DIM,
                   weights=[embeddingMatrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)
    emb1 = e0(x1)
    d=Dropout(0.2)
    o=d(emb1)
    maxpool_pool = []
    for i in range(len(filter_length)):
        conv = Conv2D(nb_filter, kernel_size=(filter_length[i], EMBEDDING_DIM),
                      kernel_initializer='he_normal', activation='elu')(o)
        maxpool_pool.append(MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_length[i] + 1, 1))(conv))

    z = concatenate(maxpool_pool)
    # z = Flatten()(z)
    z = Dropout(0.1)(z)


    # c0 = Convolution1D(nb_filter=nb_filter,
    #                         filter_length=filter_length,
    #                         border_mode=border_mode,
    #                         activation=cnn_activation,
    #                         subsample_length=1)(o)
    # p0=MaxPooling1D(pool_length=pool_length)(c0)

    lstm = Bidirectional(LSTM(dropout=0.2,output_dim=output_size))
    out = lstm(z)
    opt = Dense(NUM_CLASSES, activation='sigmoid')(out)
    dropout=Dropout(0.25)(opt)
    model = Model([x1], dropout)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


#workin on end
def build_CNNGRU_Model(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic CNN-GRU model
    """
    # Convolution parameters
    filter_length = 3
    nb_filter = 150
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(0.5))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(GRU(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('sigmoid'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('CNN-GRU')
    # model = Sequential()
    # model.add(embeddingLayer)
    # model.add(GRU(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    # model.add(Dropout(0.25))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss=loss,
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    return model

def build_LSTMCNN_Model_New(embeddingMatrix):
    filters = 250
    kernel_size = 3
    pooling = 'max'
    dropout = None
    hidden_dims = None
    lstm_units = 1800

    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input1')

    e0 = Embedding(embeddingMatrix.shape[0],
                   EMBEDDING_DIM,
                   weights=[embeddingMatrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)
    emb1 = e0(x1)
    # lstm0=LSTM(LSTM_DIM, dropout=DROPOUT,return_sequences=True)
    # out0=lstm0(emb1)
    lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))

    out = lstm(emb1)
    d=Dropout(0.25)(out)
    c0 = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(d)
    p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
    opt = Dense(NUM_CLASSES, activation='softmax')(p0)
    # final=Dense(NUM_CLASSES, activation='softmax')(opt)

    model = Model([x1], opt)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_LSTMCNN_Model(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic CNN-LSTM model

        Compiles the LSTM CNN model

        :return: compiled classifier

            """

    filters = 250
    kernel_size = 3
    pooling = 'max'
    dropout = None
    hidden_dims = None
    lstm_units = 1800

    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input1')
    # x2 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input2')
    # x3 = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32', name='main_input3')

    e0 = Embedding(embeddingMatrix.shape[0],
                   EMBEDDING_DIM,
                   weights=[embeddingMatrix],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)
    emb1 = e0(x1)
    # emb2 = e0(x2)
    # emb3 = e0(x3)

    # lstm = LSTM(lstm_units, return_sequences=True)
    # lstm = Bidirectional(LSTM(lstm_units, dropout=DROPOUT))

    # lstm1 = lstm(emb1)
    # lstm2 = lstm(emb2)
    # lstm3 = lstm(emb3)

    # inp = Concatenate(axis=-1)([emb1, emb2, emb3])

    # print(inp.shape)

    # inp = Reshape((1,lstm_units,))(inp)

    lstm = LSTM(lstm_units, return_sequences=True)
    # lstm_up = LSTM(lstm_units, dropout=DROPOUT)

    out = lstm(emb1)

    # ipt = Input(shape=(MAX_SEQUENCE_LENGTH,), sparse=False, dtype='int32')
    # opt = ipt
    #
    # opt = e0(opt)
    # opt = lstm(opt)
    p1 = []
    # # Convolution1D Layer

    for ks in range(1, kernel_size + 1):
        # Convolution1D Layer
        c0 = Conv1D(filters, ks, padding='valid', activation='relu', strides=1)(out)
        # Pooling layer:
        p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
        p1.append(p0)
    if len(p1) > 1:
        # opt = keras.concatenate(p1, axis=1)
        opt = Concatenate(axis=1)(p1)
    else:
        opt = p1[0]

    # Dropout layers
    if dropout is not None:
        d0 = Dropout(dropout)
        opt = d0(opt)

    # Output layer with sigmoid activation:
    opt = Dense(NUM_CLASSES, activation='softmax')(opt)

    # model = Model(inputs=ipt, outputs=opt)
    model = Model([x1], opt)

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_GRU_Model(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic CNN-GRU model
    """
    # Convolution parameters
    filter_length = 3
    nb_filter = 150
    pool_length = 2
    cnn_activation = 'relu'
    border_mode = 'same'

    # RNN parameters
    output_size = 50
    rnn_activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'

    # Compile parameters
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                               EMBEDDING_DIM,
                               weights=[embeddingMatrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=False)

    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(0.5))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode=border_mode,
                            activation=cnn_activation,
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(GRU(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(0.25))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('sigmoid'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('CNN-GRU')
    # model = Sequential()
    # model.add(embeddingLayer)
    # model.add(GRU(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    # model.add(Dropout(0.25))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    #
    # model.compile(loss=loss,
    #               optimizer=optimizer,
    #               metrics=['accuracy'])

    return model

def build_BiLSTMCNNwithSelfAttention_Model(wordEmbeddingMatrix):
    filters = 250
    pooling = 'max'

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    embd = Embedding(wordEmbeddingMatrix.shape[0],
                     EMBEDDING_DIM,
                     weights=[wordEmbeddingMatrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
    emb = embd(inputs)

    lstm = LSTM(LSTM_DIM, return_sequences=True)
    lstm = lstm(emb)
    print(lstm.shape)
    att = SeqSelfAttention(attention_activation='sigmoid')(lstm)

    c0 = Conv1D(filters, 1, padding='valid', activation='relu', strides=1)(att)
    p0 = {'max': GlobalMaxPooling1D()(c0), 'avg': GlobalAveragePooling1D()(c0)}.get(pooling, c0)
    print(p0.shape)
    out = Reshape((1, filters,))(p0)
    flattened = Flatten()(out)
    opt = Dense(NUM_CLASSES, activation='softmax')(flattened)

    model = Model([inputs], opt)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
        
    global trainDataPath, testDataPath, solutionPath, gloveDir, emojiDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    
    
    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    endPath = config["standard_data_path"]
    gloveDir = config["glove_dir"]
    emojiDir = config["emoji_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]
        
    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
    # writeNormalisedData(trainDataPath, trainTexts)

    trainTextsNew = []
    tweetTokenizer = TweetTokenizer()

    for i in range(len(trainTexts)):
        tokens = tweetTokenizer.tokenize(trainTexts[i])
        sent = ' '.join(tokens)
        trainTextsNew.append(sent)

    print('Size of trainText = ', len(trainTexts))
    print('Size of trainTextNew = ', len(trainTextsNew))

    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    testTextsNew = []

    for i in range(len(testTexts)):
        tokens = tweetTokenizer.tokenize(testTexts[i])
        sent = ' '.join(tokens)
        testTextsNew.append(sent)

    print('Size of testText = ', len(testTexts))
    print('Size of testTextNew = ', len(testTextsNew))

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    # stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    #               "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    #               "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    #               "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    #               "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    #               "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    #               "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    #               "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    #               "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    #               "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    #               "now"]
    # stop_words = stopwords.words('english')
    trainSequences = tokenizer.texts_to_sequences(trainTextsNew)
    # filtered_sentence_train = []
    # for w in trainSequences:
    #     if w not in stop_words:
    #         filtered_sentence_train.append(w)
    testSequences = tokenizer.texts_to_sequences(testTextsNew)
    # filtered_sentence_test = []
    # for w in testSequences:
    #     if w not in stop_words:
    #         filtered_sentence_test.append(w)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingByBin(wordIndex)
    # embeddingMatrix = getEmbeddingMatrix(wordIndex)
    # embeddingMatrix = getEmbeddingMatrixEmoji(wordIndex)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
        
    # Randomize data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]
      
    # Perform k-fold cross validation
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}

    print("\n======================================")
    
    print("Retraining model on entire data to create solution file")
    model = build_CNNLSTM_Model_Concat(embeddingMatrix)
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(testData, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

    # print("\n============= Metrics =================")
    # print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
    # print("Average Cross-Validation Micro Precision : %.4f" % (
    #         sum(metrics["microPrecision"]) / len(metrics["microPrecision"])))
    # print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"]) / len(metrics["microRecall"])))
    # print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"]) / len(metrics["microF1"])))

    print("Calculating F1 value")
    solIndices, solTexts, sollabels = preprocessData(endPath, mode="train")
    sollabels = to_categorical(np.asarray(sollabels))

    endIndices, endTexts, endlabels = preprocessData(solutionPath, mode="train")
    endlabels = to_categorical(np.asarray(endlabels))

    # predictions = model.predict(xVal, batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(endlabels, sollabels)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

if __name__ == '__main__':
    main()
