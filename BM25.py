
# coding: utf-8

# In[1]:


import math
import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords


# In[2]:


#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0
stop_words = set(stopwords.words('english'))
stemma = nltk.PorterStemmer()
lemma = nltk.WordNetLemmatizer()

print("Variable Defined")


# In[33]:


def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    i = 0
    for line in f:
        i+=1
        if(i%50000 == 0):
            print(i)
        passage = line.strip().lower().split("\t")[2]
        fw.write(passage+"\n")
    f.close()
    fw.close()
    print("Corpus Done")


# In[41]:


# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula 
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0
    
    for line in open(corpusfile,"r",encoding="utf-8"):
        doc = (''.join([j.lower() for j in line if j in string.ascii_letters + ' '])).split(delimiter)
        t_doc = []
        for i in doc:
            if i not in stop_words:
#                 t_doc.append(stemma.stem(i))
#                 t_doc.append(lemma.lemmatize(i))
                t_doc.append(i)
        doc = t_doc
        totalDocLength += len(doc)
        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%50000==0):
            print(numOfDocuments)                

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = 1 - math.log(docFrequencyDict[word])/(1+ numOfDocuments)

    avgDocLength = totalDocLength / numOfDocuments

     
    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()


    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)


# In[42]:


#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' '):
    global docIDFDict,avgDocLength

    query_words= (''.join([j.lower() for j in Query if j in string.ascii_letters + ' '])).split(delimiter)
    t_doc = []
    for i in query_words:
        if i not in stop_words:
            t_doc.append(i)
    query_words = t_doc

    passage_words = (''.join([j.lower() for j in Passage if j in string.ascii_letters + ' '])).split(delimiter)
    t_doc = []
    for i in passage_words:
        if i not in stop_words:
            t_doc.append(i)
    passage_words = t_doc
    passageLen = len(passage_words)

    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score


# In[22]:


#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):
    NumOfDocuments :  4717692
    AvgDocLength :  33.955410187863045
    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%50000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()


# In[23]:


inputFileName = "traindata.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
testFileName = "validationdata.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
corpusFileName = "corpus.tsv" 
outputFileName = "answer_BM.tsv"


# In[ ]:


GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
print("Corpus File is created.")


# In[43]:


IDF_Generator(corpusFileName)   # Calculates IDF scores. 
print("IDF Dictionary Generated.")


# In[44]:


RunBM25OnEvaluationSet(testFileName,outputFileName)
print("Submission file created. ")

