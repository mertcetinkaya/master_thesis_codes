import tsr_lib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
#%%
#german data examples with hog feature

tsr_lib.produce_training_data([5,10,15],40,'gtsrb-german-traffic-sign',43)
tsr_lib.produce_test_data([5,10,15],40,'gtsrb-german-traffic-sign')
tsr_lib.function('lr',[5,10,15],43,1,'gtsrb-german-traffic-sign')


#%%


def show_statistics(records,titles):
    col_names=[]
    for title in titles:
        col_names.append(title)
    row_names=['nobs','min','max','mean','median','std','variance']
    frame=pd.DataFrame(index=row_names,columns=col_names)
    for i in range(len(records)):
        stats=[len(records[i]),np.min(records[i]),np.max(records[i]),np.mean(records[i]),np.median(records[i]),np.std(records[i]),np.var(records[i])]
        frame.iloc[:,i]=stats
        
    return frame

#%%

#german data examples with cnn
#germany
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
data=tsr_lib.cnn_produce_data('gtsrb-german-traffic-sign',edge_length,43)

model_count=2
for i in range(1,model_count+1):  
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(43,activation='softmax'))
    learning_rate=0.001
    epochs=5
    decay_rate = learning_rate / epochs
    opt = Adam(lr=0.001, decay=learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    a,b,c,d,e=tsr_lib.cnn(model,edge_length,epochs,43,'gtsrb-german-traffic-sign',i,training_val_test_data=data)
    members.append(model)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    
  


fr=show_statistics([training_acc,val_acc,test_acc],['training','val','test'])


plt.plot(range(1,len(training_acc)+1),training_acc,'k.-')
plt.plot(range(1,len(training_acc)+1),val_acc,'b.-')
plt.xticks(np.arange(1,model_count+1,1))
plt.legend(labels=['training_acc','val_acc'])
plt.ylabel('accuracy')
plt.xlabel('model')
plt.title('accuracy by model')
plt.savefig('gtsrb-german-traffic-sign/acc_by_model.jpg')
plt.show()

#%%

plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=False)
#%%

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, weights, X):
    # make predictions
    yhats = [model.predict(X) for model in members]
    #yhats = [to_categorical(model.predict_classes(testX)) for model in members]
    yhats = np.array(yhats)
    # weighted sum across ensemble members
    summed = np.tensordot(yhats, weights, axes=((0),(0)))
    # argmax across classes
    result = np.argmax(summed, axis=1)
    return result

# # evaluate a specific number of members in an ensemble
def evaluate_ensemble(members, weights, X, y):
    # make prediction
    yhat = ensemble_predictions(members, weights, X)
    # calculate accuracy
    return accuracy_score(y, yhat)

# normalize a vector to have unit norm
def normalize(weights):
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result
#%%
weights = [1.0/2 for _ in range(2)]
print('weights: ',weights)
score = evaluate_ensemble(members, weights, data[1], np.argmax(data[4],axis=1))
print('Equal Weights Score: ', score, ' val')
score = evaluate_ensemble(members, weights, data[2], data[5])
print('Equal Weights Score: ', score, ' test')

#%%

weights = list(val_acc/sum(val_acc))
print('weights: ',weights)
score = evaluate_ensemble(members, weights, data[1], np.argmax(data[4],axis=1))
print('Weighted Score: ', score, ' val')
score = evaluate_ensemble(members, weights, data[2], data[5])
print('Weighted Score: ', score, ' test')

#%%

from itertools import product
def grid_search(members, valX, valy, testX, testy):
    # define weights to consider
    w = [0.0, 0.5, 1.0]
    duplicate_weights_list=[list(normalize(i)) for i in product(w, repeat=len(members))]
    weights_list=[]
    for i in duplicate_weights_list:
        if not(i in weights_list):
            weights_list.append(i)
    best_score, best_weights = 0.0, None
    # iterate all possible combinations (cartesian product)
    print('weights list len: ',len(weights_list))
    ctr=0
    for weights in weights_list:
        # skip if all weights are equal
        if (weights == len(members)*[0.0]):
            continue
        # hack, normalize weight vector
        # evaluate weights
        score = evaluate_ensemble(members, weights, valX, valy)
        if score > best_score:
            best_score, best_weights = score, weights
            print(best_weights ,' val: ', best_score, 'test: ', evaluate_ensemble(members, weights, testX, testy))
        ctr+=1
        if(ctr%1==0):
            print('evaluated weights number: ',ctr)
    return list(best_weights)

#%%
    
weights = grid_search(members, data[1], np.argmax(data[4],axis=1),data[2], data[5])
val_score = evaluate_ensemble(members, weights, data[1], np.argmax(data[4],axis=1))
test_score = evaluate_ensemble(members, weights, data[2], data[5])
print('Grid Search Weights: ', weights, 'Val Score: ' , val_score, 'Test Score: ', test_score)
#%%

yhat = ensemble_predictions(members, weights, data[2])
# calculate accuracy
accuracy_score(data[5], yhat)
#%%
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
cr=classification_report(data[5], yhat,digits=4)
fig = plt.figure(figsize=(20, 20))
sns.heatmap(confusion_matrix(y_pred=yhat,y_true=data[5],labels=range(43)),  annot=True,fmt='g',annot_kws={"size":8})
plt.title('test confusion matrix')
plt.ylabel('actual')
plt.xlabel('predicted')
#plt.savefig(traffic_sign_directory+'/test_conf'+str(model_number)+'.jpg')
plt.show()



#%%
#belgium data examples with cnn
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]

data=tsr_lib.cnn_produce_data('btsc',edge_length,62)

model_count=1
for i in range(1,model_count+1):    
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(512, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
         
    
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(62,activation='softmax'))
    learning_rate=0.001
    epochs=5
    decay_rate = learning_rate / epochs
    opt = Adam(lr=0.001, decay=learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    a,b,c,d,e=tsr_lib.cnn(model,edge_length,epochs,62,'btsc',i,training_val_test_data=data)
    members.append(model)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    
#%%
from sklearn.metrics import classification_report
cr=classification_report(data[5],members[0].predict_classes(data[2]),digits=4)
    
#%%

fr=show_statistics([training_acc,val_acc,test_acc],['training','val','test'])

model_count=1
plt.plot(range(1,len(training_acc)+1),training_acc,'k.-')
plt.plot(range(1,len(training_acc)+1),val_acc,'b.-')
plt.xticks(np.arange(1,model_count+1,1))
plt.legend(labels=['training_acc','val_acc'])
plt.ylabel('accuracy')
plt.xlabel('model')
plt.title('accuracy by model')
plt.savefig('btsc/acc_by_model.jpg')
plt.show()


