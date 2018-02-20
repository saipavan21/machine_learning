
#import
import time
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

#stopwords
stopWords = set(stopwords.words('english'))

#data import
train_data = pd.read_csv('reviews_tr.csv')
test_data = pd.read_csv('reviews_te.csv')

#get train and test arrays
count_vec = CountVectorizer(stop_words= stopWords, min_df = 10)
X_train = count_vec.fit_transform(train_data.text)
count_vec_test = CountVectorizer(stop_words= stopWords)
X_test = count_vec_test.fit_transform(test_data.text)
Y_train = train_data.label.replace(0,-1)
Y_test = test_data.label.replace(0,-1)

#training function
def get_weights(X_df, Y_train):
    W = np.zeros((1,X_df.shape[1]+1))
    x_idx = np.arange(X_df.shape[0])
    for epoch in range(2):
        np.random.shuffle(x_idx)
        count = 0
        tot_W = W
        for i in x_idx:
            x_train = X_df[i]
            x_train = hstack([x_train,[[1]]])
            value = Y_train[i] * x_train.dot(W.T)
            if value <= 0:
                W = W + (Y_train[i] * x_train)
                
            if epoch == 1:
                tot_W += W

            count += 1
            if count%10000 ==0:
                print("In epoch {} and completd {} data points".format( epoch, count))
                
        print("completed epoch")

    return tot_W/(X_df.shape[0]+1)

#Accuracy function
def get_accuracy(X, Y, W):
    count = 0
    for i in range(X.shape[0]):
        x_val = X[i]
        x_val = hstack([x_val,[[1]]])
        y_pred = x_val.dot(W.T)
        #print(y_pred)
        if y_pred[0] > 0:
            y_pred = 1
        else:
            y_pred[0] = -1
        if y_pred == Y[i]:
            count += 1.0

    acc = count/X.shape[0]
    return(acc)

#training the model
starttime = time.time()
tf_W = get_weights(X_train, Y_train)
print("Training took {} seconds".format(time.time()-starttime))

#Saving the model
train_feat_df = pd.DataFrame.from_dict(count_vec.vocabulary_, orient= 'index').sort_values(by=[0])
train_feat_df['feat'] = train_feat_df.index
train_feat_df = train_feat_df.reset_index(drop=True)
temp_train_df = pd.DataFrame([[train_feat_df.shape[0],'bias1']], columns = [0, 'feat'])
train_feat_df = train_feat_df.append(temp_train_df, ignore_index=True)
train_feat_df['weight'] = tf_W.T
train_feat_df.to_csv('unigram_W.csv', sep=',',index=False)

#Training Accuracy
train_acc = get_accuracy(X_train, Y_train, tf_W)
print("Unigram training Accuracy:",train_acc)

#Test feature vector mapping
test_feat_df = pd.DataFrame.from_dict(count_vec_test.vocabulary_, orient = 'index').sort_values(by=[0])
test_feat_df['feat'] = test_feat_df.index
test_feat_df = test_feat_df.reset_index(drop=True)
temp_test_df = pd.DataFrame([[test_feat_df.shape[0],'bias1']], columns = [0, 'feat'])
test_feat_df = test_feat_df.append(temp_test_df, ignore_index=True)
tf_test_W = pd.merge(test_feat_df, train_feat_df, on= 'feat', how='left').fillna(0)['weight']

test_acc = get_accuracy(X_test, Y_test, tf_test_W)
print("Unigram test accuracy:",test_acc)