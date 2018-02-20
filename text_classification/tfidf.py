
#import
import time
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

#data import
train_data = pd.read_csv('reviews_tr.csv')
test_data = pd.read_csv('reviews_te.csv')

#train and test arrays
count_vec = CountVectorizer()
X_train = count_vec.fit_transform(train_data.text)
count_vec_test = CountVectorizer()
X_test = count_vec_test.fit_transform(test_data.text)
Y_train = train_data.label.replace(0,-1)
Y_test = test_data.label.replace(0,-1)

#data preprocess
def get_filtered_words(sentence):
    word_tokens = sentence.split(" ")
    return word_tokens

train_data['words'] = train_data.text.apply(lambda x: get_filtered_words(x))

#feature idf scores
total_list = []
for idx, row in train_data.iterrows():
    total_list += list(set(row['words']))

tot_df = pd.DataFrame.from_dict(Counter(total_list), orient='index')
tot_df['idf'] = np.log10(train_data.shape[0]/tot_df[0])
tot_df['feat'] = tot_df.index

vocab_df = pd.DataFrame.from_dict(count_vec.vocabulary_, orient='index').sort_values(by=0)
vocab_df['feat'] = vocab_df.index
idf_vec = pd.merge(vocab_df, tot_df, on= 'feat', how='left').fillna(0)['idf']
idf_vec = np.reshape(np.array(idf_vec), (1, idf_vec.shape[0]))

#Training function
def get_weights(X_df, Y_train, idf_vec):
    W = np.zeros((1,X_df.shape[1]+1))
    x_idx = np.arange(X_df.shape[0])
    for epoch in range(2):
        np.random.shuffle(x_idx)
        count = 0
        tot_W = W
        for i in x_idx:
            x_train = X_df[i]
            x_train = x_train.multiply(idf_vec)
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


#accuracy function
def get_accuracy(X, Y, W, idf_score):
    count = 0
    for i in range(X.shape[0]):
        x_val = X[i]
        x_val = x_val.multiply(idf_score)
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
    return acc

#Model training
starttime = time.time()
tf_idf_W = get_weights(X_train, Y_train, idf_vec)
print("Tfidf training took {} seconds".format(time.time()-starttime))

#Training Accuracy
train_acc = get_accuracy(X_train, Y_train, tf_idf_W, idf_vec)
print("TfIdf training accuracy:",train_acc)

#Test feature vector mapping
vocab_test_df = pd.DataFrame.from_dict(count_vec_test.vocabulary_, orient='index').sort_values(by=0)
vocab_test_df['feat'] = vocab_test_df.index
test_idf = pd.merge(vocab_test_df, tot_df, on= 'feat', how='left').fillna(0)['idf']
test_idf = np.reshape(np.array(test_idf), (1, test_idf.shape[0]))
temp_test_df = pd.DataFrame([[vocab_test_df.shape[0],'bias1']], columns = [0, 'feat'])
vocab_test_df = vocab_test_df.append(temp_test_df, ignore_index=True)
test_W = pd.merge(vocab_test_df, vocab_df, on= 'feat', how='left').fillna(0)['weight']

#Test Accuracy
test_acc = get_accuracy(X_test, Y_test, test_W, test_idf)
print("TfIdf test accuracy:",test_acc)