#Vector operations and data management
import pandas as pd
import scipy
import numpy as np

#Folder management
import os,sys,inspect

#Model and other data type saving
import pickle

#Printing styling
import pprint
from IPython.display import display, HTML

#Plot management
import seaborn as sns
import matplotlib.pyplot as plt

#Skseq
import skseq
from skseq.sequences import sequence
from skseq.sequences.sequence import Sequence
from skseq.sequences.sequence_list import SequenceList
from skseq.sequences.label_dictionary import LabelDictionary
import skseq.sequences.structured_perceptron as spc
from skseq.sequences import extended_feature

#Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Accuracy computing
def accuracy(model,data,dic_tag_pos,corpus_tag_dict):
    
    total = 0.0
    correct = 0.0
    correct_sentence = 0.0
    y_pred = []
    
    #For every sentence
    for i in range(len(data)):
        #Get the predicted tags
        pred_tags = model.predict_tags_given_words(data[i])
        y_pred.append(pred_tags)
        #For each predicted tag
        for index,tag_pos in enumerate(pred_tags):
            #If both predicted & true tags are "O": ignore
            #If both tags are equal and not "O", count as a correct prediction
            if tag_pos == dic_tag_pos[i][index] and tag_pos != corpus_tag_dict["O"]:
                correct += 1
                #Count the tag
                total += 1
            #When there has been a miss-prediction, sum up the total number of tags
            elif tag_pos != dic_tag_pos[i][index]:
                total += 1
        #Check if the whole sentence is exactly the same (including "O" tags)
        if list(pred_tags) == dic_tag_pos[i]:
            correct_sentence += 1
    return y_pred, correct_sentence, correct, total

#Gets the accuracy and computes the other metrics
def evaluate(model, data, dic_tag_pos, corpus_tag_dic,y_true,save_name,dataframe=None):

    #If we do not pass a dataframe means we are evaluating the model for the first time
    if dataframe is None:
        #Get the info to compute the metrics
        list_y_pred, correct_sentence, correct, total = accuracy(model,data, dic_tag_pos, corpus_tag_dic)
        y_pred = [tag for array in list_y_pred for tag in array]
        #Save y_pred
        save("fitted_models/y_pred_"+save_name+".pkl",y_pred)
        
        #Save a dataframe storing the total number of correct sentences, the accuracy and the F1 Weighted Score
        dataframe = pd.DataFrame({"Correct Sentences":[correct_sentence/len(data)],
                         "Correct Tags":[correct/total],
                          "Weighted F1 Score":[f1_score(y_true,y_pred,average="weighted")]})
        
        dataframe.to_csv("Results/"+save_name+".csv",index=None)
    
    #Re-getting y_pred so I do not have to code an if
    y_pred = load("fitted_models/y_pred_"+save_name+".pkl")
    
    print("01. Metrics:")
    display(HTML(round(dataframe*100,2).to_html()))

    print("\n02. Confusion matrix:")
    #When executing this function with train and tests sets, model.state_labels is the dict 
    #With all the possible tags. Nevertheless, for TINY_TEST, not all the tags in y_true are in 
    #model.state_lables. Therefore, we are creating this feasible tag dict for the Confusion matrix tick labels. 
    feasible_dict = {k:corpus_tag_dic[k] for k in [{v: k for k, v in corpus_tag_dic.items()}[i] for i in set(y_true+y_pred)] if k in corpus_tag_dic}
    plot_confusion_matrix(y_true,y_pred,feasible_dict)

        
    return y_pred

def plot_confusion_matrix(y_true,y_pred,tag_names):
    
    #Get the tags names
    labels = [i for i in tag_names]
    #Plot size
    fig, ax = plt.subplots(figsize=(20,20))
    #NumPy confusion matrix for each class
    cm = confusion_matrix(y_true, y_pred)

    #Seaborn heatmap call
    ax = sns.heatmap(cm, annot=True, ax=ax,fmt="d", linewidths=0, cmap = 'viridis', xticklabels = True)
    #Axis beautify
    ax.set(ylabel='True Tag',xlabel='Predicted Tag',xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels)
    
    plt.show()
    
    print("\n03. Metrics by tag:")
    display(HTML(pd.DataFrame(classification_report(
                y_true,y_pred,output_dict=True)).transpose().drop(["support"],axis=1).rename(
                index={str(v): k for k, v in tag_names.items()}).to_html()))


#Returns the TINY TEST dataset, including our own ground truth tags.
def get_tiny_test():
    TINY_TEST = [["The programmers from Barcelona might write a sentence without a spell checker . "],
    ["The programmers from Barchelona cannot write a sentence without a spell checker . "],
    ["Jack London went to Parris . "],
    ["Jack London went to Paris . "],
    ["Bill gates and Steve jobs never though Microsoft would become such a big company . "],
    ["Bill Gates and Steve Jobs never though Microsof would become such a big company . "],
    ["The president of U.S.A though they could win the war . "],
    ["The president of the United States of America though they could win the war . "],
    ["The king of Saudi Arabia wanted total control . "],
    ["Robin does not want to go to Saudi Arabia . "],
    ["Apple is a great company . "],
    ["I really love apples and oranges . "],
    ["Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . "]]

    TAGS = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per','O','O','I-org','O','O','O','O','O','O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per','O','O','I-org','O','O','O','O','O','O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per','O','O','O','O','O','O','B-geo', 'I-geo','O'],
            ['B-org','O','O','O','O','O'],
            ['O','O','O','O','O','O','O'],
            ['B-per','O','B-per','O','O','O','B-org','O','O','O','O','O','O','O','O','O','O','B-geo','I-geo','O']]


    return [i[0].split() for i in TINY_TEST],TAGS

#Function that creates two dictionaries:
#word_pos: each unique word (key) is assigned to a unique index (value)
#tag_pos: each unique tag (key) is assigned to a unique index (value)
def corpus(X_train, y_train):
    i = 0
    word_pos_dict = {}
    for sentence in X_train:
        for word in sentence:
            if word not in word_pos_dict:
                word_pos_dict[word] = i
                i+=1
    i = 0
    tag_pos_dict = {}
    for sentence in y_train:
        for tag in sentence:
            if tag not in tag_pos_dict:
                tag_pos_dict[tag] = i
                i +=1
                
    return word_pos_dict, tag_pos_dict

#Pickle loading / saving helpers
def save(name,file):
    with open(name, "wb") as f:
        pickle.dump(file, f)
def load(name):
    with open(name, 'rb') as f:
        return(pickle.load(f))