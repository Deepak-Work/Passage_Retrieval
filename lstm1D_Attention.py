import re
import time
import numpy as np
from numpy.random import randint
from keras.utils import to_categorical
import keras
from keras.layers.core import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import merge, Conv1D, MaxPooling1D, Input, Dense, multiply, concatenate, Flatten, ConvLSTM2D, Convolution1D,LSTM, Reshape,Bidirectional, Concatenate
from keras.utils import to_categorical
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import LearningRateScheduler
from layers import Addition, AttentionWithContext

warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#Initialize Global variables 
GloveEmbeddings = {}
GloveEmbeddingsother = {}
max_query_words = 12
max_passage_words = 50
num_classes = 2
emb_dim = 100
emb_dim_other = 100
validation_query_vectors = []
validation_passage_vectors = []
validation_labels = []   
q_max_words=12
p_max_words=50
lr=0.005

#The following method takes Glove Embedding file and stores all words and their embeddings in a dictionary
def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddings[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = "0.0 "*emb_dim
    fe.close()

def loadotherEmbeddings(embeddingfile):
    global GloveEmbeddingsother,emb_dim_other

    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddingsother[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddingsother["zerovec"] = "0.0 "*emb_dim_other
    fe.close()


def split(end):    
    file_data = open("data.tsv", encoding="utf8")    

    i = 0
    while(True):
        a = (file_data.readline())
        if (a == ''):
            break
        i+=1    

    count = int(i*end)    

    split = int((.98*count)) - int(((.98*count)%10)) #Split the data into end-2% train data 2% validation data and the rest to test data.
    print(count)
    print(split)
    file_data.close()    

    file_data = open("data.tsv", encoding="utf8")
    file_train = open("traindata.tsv", "w+", encoding="utf8")
    file_validation = open("validationdata.tsv", 'w+', encoding="utf8")    

    for i in range(0,split):
        file_train.write(file_data.readline())    

    for i in range(split,count):
        file_validation.write(file_data.readline())


def TextDataToarray(inputfile,isEvaluation,outputfile):
    final_array = []
    global GloveEmbeddings,GloveEmbeddingsother,emb_dim,max_query_words,max_passage_words
    i = 0
    f = open(inputfile,"r",encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    f_out = open(outputfile,"w+",encoding="utf-8",errors="ignore")

    for line in f:
        i+=1
        if(i%50000==0):
            print(i)
        tokens = line.strip().lower().split("\t")
        query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]

        #****Query Processing****
        words = re.split('\W+', query)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_query_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words] # trim extra words
        #create Query Feature vector 
        query_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddingsother):
                query_feature_vector += GloveEmbeddingsother[word]+" "
            else:
                query_feature_vector += GloveEmbeddingsother["zerovec"]+" "  #Add zerovec for OOV terms
        query_feature_vector = query_feature_vector.strip() 

        #***** Passage Processing **********
        words = re.split('\W+', passage)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_passage_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words] # trim extra words
        #create Passage Feature vector 
        passage_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddings):
                passage_feature_vector += GloveEmbeddings[word]+" "
            else:
                passage_feature_vector += GloveEmbeddings["zerovec"]+" "  #Add zerovec for OOV terms
        passage_feature_vector = passage_feature_vector.strip() 


        if(not isEvaluation):
            f_out.write(query_feature_vector+'|'+passage_feature_vector+'|'+label+'\n')
        else:
            f_out.write(query_feature_vector+'|'+ passage_feature_vector+'|'+ (query_id)+'\n')

#Sampling for more balanced train dataset
def sampling(testfile, finalfile):
    f = open(testfile,'r',encoding="utf-8")
    f_out = open(finalfile, 'w+', encoding='utf-8')

    for line in f:
        token = line.strip().split("|")
        if(int(token[2]) ==  1):
            f_out.write(line)
        elif(int(token[2]) == 0 and randint(0, high = 3) == 1):
            f_out.write(line)

## The following GetPredictionOnEvalSet method reads all query passage pair vectors from CTF file and does forward prop with trained model to get similarity score
## after getting scores for all the pairs, the output will be written into submission file. 
def GetPredictionOnanySet(model,testfile,submissionfile,batch_size):
    global q_max_words,p_max_words,emb_dim
    count = 0
    file_array_query = np.zeros((batch_size, max_query_words, emb_dim_other))
    file_array_passage = np.zeros((batch_size, max_passage_words, emb_dim))
    query_id_array = np.zeros((batch_size, 1))

    f = open(testfile,'r',encoding="utf-8")
    all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
    for line in f:

        tokens = line.strip().split("|")  
        x1 = tokens[0].strip() #Query Features
        x2 = tokens[1].strip() # Passage Features
        query_id = tokens[2].replace("qid","").strip() # Query_id
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]    
        queryVec   = np.array(x1,dtype="float32").reshape(1,q_max_words,emb_dim_other)
        passageVec = np.array(x2,dtype="float32").reshape(1,p_max_words,emb_dim)
        file_array_query[count] = queryVec
        file_array_passage[count] = passageVec
        query_id_array[count] = query_id
        count+=1

        if(count == batch_size):
            score = model.predict([file_array_query,file_array_passage]) # do forward-prop on model to get score\
            
            for i in range(0, count):
                if(query_id_array[i][0] in all_scores):
                    all_scores[query_id_array[i][0]].append(score[i][1])
                else:
                    all_scores[query_id_array[i][0]] = [score[i][1]]
            count = 0
    if (count != 0):
        score = model.predict([file_array_query,file_array_passage])

        for i in range(0, count):
            if(query_id_array[i][0] in all_scores):
                all_scores[query_id_array[i][0]].append(score[i][1])
            else:
                all_scores[query_id_array[i][0]] = [score[i][1]]

    fw = open(submissionfile,"w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
        fw.write(str(query_id)+"\t"+scores_str+"\n")
    fw.close()


def generate(file_name, isEvaluation, batch_size):
    i = 0
    file_array_query = np.zeros((batch_size, max_query_words, emb_dim_other))
    file_array_passage = np.zeros((batch_size, max_passage_words, emb_dim))
    label_list = np.zeros((batch_size,2))
    
    while True:
        with open(file_name,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                    
                tokens = line.strip().split("|")
                
                x1 = tokens[0].strip()
                query_feature_vector = [float(v) for v in x1.split()]
                query_array = np.array(query_feature_vector)
                q_array = np.reshape(query_array,(max_query_words, emb_dim_other))
                

                x2 = tokens[1].strip()
                passage_feature_vector = [float(v) for v in x2.split()]
                passage_array = np.array(passage_feature_vector)
                p_array = np.reshape(passage_array,(max_passage_words, emb_dim))
                x3 = float(tokens[2].strip())

                file_array_query[i] = q_array
                file_array_passage[i] = p_array
                if(x3 == 0):
                    label_list[i][0] = 1
                    label_list[i][1] = 0
                else:
                    label_list[i][1] = 1
                    label_list[i][0] = 0
                i+=1
                if( i >= batch_size):
                    i = 0
                    yield ([file_array_query, file_array_passage], (label_list))


def TrainAndValidate():

    #*****Hyper-Parameters******
    q_max_words= 12
    p_max_words = 50
    emb_dim = 100
    hidden_dim = 64
    num_classes = 2
    batch_size = 250
    epoch_size = 100000 #No.of samples in training set
    total_epochs = 50 #Total number of epochs to run
    query_total_dim = q_max_words*emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words*emb_dim

    #behold the model is here
    #input for training data
    query = Input(shape=(12,100), dtype='float32')
    passage = Input(shape=(50,100), dtype='float32')
    # pass the question embedding through bi-lstm
    
    passage_bidir_encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True,dropout=0.2, kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'), merge_mode='concat')
    query_bidir_encoder = Bidirectional(LSTM(hidden_dim, return_sequences=True,dropout=0.2, kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'), merge_mode='concat')

    encoded_passage = passage_bidir_encoder(passage)
    encoded_question = query_bidir_encoder(query)
    tiled_passage = Lambda(lambda x: tf.tile(tf.expand_dims(x, 2), [1, 1, q_max_words, 1]))(encoded_passage)
    tiled_question = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, p_max_words, 1, 1]))(encoded_question)

    #Adding BIDAF model
    a_elmwise_mul_b = Lambda(lambda x:tf.multiply(x[0], x[1]))([tiled_passage, tiled_question])

    cat_data = Concatenate()([tiled_passage, tiled_question, a_elmwise_mul_b])

    S = Dense(1)(cat_data)
    S = Lambda(lambda x: K.squeeze(x, -1))(S)  

    S = Activation('softmax')(S)

    c2q = Lambda(lambda x: tf.matmul(x[0], x[1]))([S, encoded_question]) 

    # Query2Context
    # b: attention weights on the context
    b = Lambda(lambda x: tf.nn.softmax(K.max(x, 2), dim=-1), name='b')(S)

    q2c = Lambda(lambda x:tf.matmul(tf.expand_dims(x[0], 1), x[1]))([b, encoded_passage])
    q2c = Lambda(lambda x: tf.tile(x, [1, max_passage_words, 1]))(q2c) 


    # G: query aware representation of each context word
    G = Lambda(lambda x: tf.concat([x[0], x[1], tf.multiply(x[0], x[1]), tf.multiply(x[0], x[2])], axis=2)) ([encoded_passage, c2q, q2c]) 
    

    modelled_passage = Bidirectional(LSTM(hidden_dim, return_sequences=True))(G)
    modelled_passage = Bidirectional(LSTM(hidden_dim, return_sequences=True))(modelled_passage)

    g2 = Lambda(lambda x: tf.reduce_sum(x, 2))(modelled_passage)
    output = Dense(2, activation='softmax')(g2)
    
    model = Model(inputs=[query, passage], outputs=output)
    adam = keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy')

    return model


if __name__ == "__main__":

    start = time.time()
    print("Code started")
    loadEmbeddings("glove.6B.100d.txt")
    print("100 embeddings loaded")
    loadotherEmbeddings("glove.6B.100d.txt")
    print("100 other embeddings loaded")
    split(0.9) #The ratio in which data.tsv is to be split in training and validation data
    print("Split done")
    sampling("traindata.tsv","train_sampled.tsv") #Sampling each query in dataset with 1 positive and 3 negatives.
    print("Sampling done")
    TextDataToarray("train_sampled.tsv", False, "train_inter.tsv")
    print("Train converted")
    model = TrainAndValidate() # Training and validation methods
    print("Model created")
    print(model.summary())

    checkpointer = ModelCheckpoint("my_checkpoint.h5", verbose=1, save_best_only=False)
    model.fit_generator(generate("train_inter.tsv", False, 250), steps_per_epoch = 400, epochs = 90, callbacks=[checkpointer])
    model.save('my_model.h5')
    TextDataToarray("validationdata.tsv", True, "validout.tsv")
    print("Test Converted")
    #model = load_model("my_model.h5", custom_objects ={'tf':tf, 'max_passage_words':max_passage_words})
    #print("Model loaded")
    print("Predicting")
    GetPredictionOnanySet(model, "validout.tsv", "answer.tsv",500)
    print(time.time() - start)
