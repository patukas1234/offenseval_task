import numpy as np
import pandas as pd

from utils import *
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils



def pipe(preproc_train_data, preproc_test_data, encoded_train_labels, encoded_test_labels, n_range):
    
    train_ngram, test_ngram = get_ngrams(n_range, preproc_train_data, preproc_test_data)

    predictions = SVM(train_ngram, encoded_train_labels,test_ngram)


    f1, precision, recall, accuracy = get_metrics(predictions, encoded_test_label)

    dataset = pd.DataFrame({'F1': [f1], 'Precision': [precision], 'Recall' : [recall], 'Accuracy' : [accuracy]})

    return dataset





def get_ngrams(n_range, data_train, data_test):
    
    word_vectorizer = CountVectorizer(ngram_range= n_range) # change 1,1 to n_range
    
    ngram_data = word_vectorizer.fit_transform(data_train)
    
    ngram_data = ngram_data.astype(float)

    ngram_data_test = word_vectorizer.transform(data_test)
    ngram_data_test = word_vectorizer.transform(data_test)
    
    return ngram_data, ngram_data_test



def SVM(train_data, train_labels, test_data):
    max_abs_scaler = MaxAbsScaler()

    scaled_train_data = max_abs_scaler.fit_transform(train_data)
    scaled_test_data = max_abs_scaler.transform(test_data)
    svm_clf= SVC(C=1)  #TODO: add parameter for the C regularization value 
    svm_clf.fit(scaled_train_data, train_labels)
    predictions = svm_clf.predict(scaled_test_data)

    return predictions




def get_metrics(predicted, actual): 
    f1 = f1_score(predicted, actual, average="weighted")
    
    precision = precision_score(predicted, actual, average="weighted")
    
    recall = recall_score(predicted, actual, average="weighted")

    accuracy = accuracy_score(predicted, actual)

    return f1, precision, recall, accuracy


#def get_baseline_model_fn(num_in, num_out):
    # for Keras: initialize baseline model fn with num features and num predicted categories
 #   def baseline_model():
        # create model
  #      model = Sequential()
   #     model.add(Dense(8, input_dim=num_in, activation='relu'))
    #    model.add(Dense(num_out, activation='softmax'))
        # Compile model
     #   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      #  return model
    #return baseline_model








