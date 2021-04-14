import tsr_lib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.layers import TimeDistributed
from keras.layers import LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
#%%

#tsr_lib.produce_training_data([4,8,12],32,'gtsrb-german-traffic-sign',43)
tsr_lib.produce_test_data([4,8,12],32,'gtsrb-german-traffic-sign')

#%%
#german
edge_length=32
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
#lr = 0.01
#from keras.optimizers import SGD
#sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tsr_lib.cnn([model],edge_length,2,43,'gtsrb-german-traffic-sign')

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


#germany
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
#data=tsr_lib.cnn_produce_data('GTSRB',edge_length,43)
#data=tsr_lib.cnn_produce_data('btsc',edge_length,62)
data=tsr_lib.cnn_produce_data('gtsrb-german-traffic-sign',edge_length,43)

model_count=2
for i in range(1,model_count+1):  
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu',padding='same')) #padding yok
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu',padding='same')) #padding yok
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # model.add(Conv2D(128, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
      
    
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(43,activation='softmax'))
    #model.add(Dense(62,activation='softmax'))
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    learning_rate=0.001
    epochs=25
    decay_rate = learning_rate / epochs
    opt = Adam(lr=0.001, decay=learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #a,b,c,d,e=tsr_lib.cnn(model,edge_length,25,43,'GTSRB',i,training_val_test_data=data)
    a,b,c,d,e=tsr_lib.cnn(model,edge_length,25,43,'gtsrb-german-traffic-sign',i,training_val_test_data=data)
    #a,b,c,d,e=tsr_lib.cnn(model,edge_length,25,62,'btsc',i,training_val_test_data=data)
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
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
# model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu',padding='same')) #padding yok
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Flatten())
# model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
# model.add(Dense(43,activation='softmax'))
from keras.utils.vis_utils import plot_model
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
e_norm=e/max(e)
inp=1+e_norm
coeff=d/inp
class_weight_list=np.ones(len(e))/coeff-e_norm
class_weight_list=class_weight_list/min(class_weight_list)
max_value=max(class_weight_list[class_weight_list!=np.inf])
for i in range(len(class_weight_list)):
    if(class_weight_list[i]==np.inf):
        class_weight_list[i]=max_value
class_weight=dict(zip(range(62),class_weight_list)) 



#%%

#model.save('my_model.h5')
del model
from keras.models import load_model
model = load_model('my_model.h5')
#%%
#belgium
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
#data=tsr_lib.cnn_produce_data('gtsrb-german-traffic-sign',edge_length,43)
data=tsr_lib.cnn_produce_data('btsc',edge_length,62)
#data=tsr_lib.cnn_produce_data('GTSRB',edge_length,43)

model_count=1
for i in range(1,model_count+1):    
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    
    model.add(Conv2D(512, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
#    model.add(Conv2D(128, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu'))
#    model.add(BatchNormalization())
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
      
    
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    #model.add(Dense(43,activation='softmax'))
    model.add(Dense(62,activation='softmax'))
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    learning_rate=0.001
    epochs=3
    decay_rate = learning_rate / epochs
    opt = Adam(lr=0.001, decay=learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #a,b,c=tsr_lib.cnn(model,edge_length,epochs,43,'GTSRB',training_val_test_data=data)  
    a,b,c,d,e=tsr_lib.cnn(model,edge_length,2,62,'btsc',i,training_val_test_data=data)
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
#%%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=False)

#%%
import pandas as pd
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
#belgium
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
data=tsr_lib.cnn_produce_data('btsc',edge_length,62)
for i in range(5):  
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',kernel_initializer='he_uniform', input_shape=(edge_length,edge_length,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
      
    
    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='he_uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(62,activation='softmax'))
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    a,b,c=tsr_lib.cnn(model,edge_length,10,62,'btsc',training_val_test_data=data)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    


fr=show_statistics([training_acc,val_acc,test_acc],['training','val','test'])

#%%
#transfer learning
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications import MobileNet
edge_length=32
base_model=MobileNet(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(256,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
preds=Dense(43,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tsr_lib.cnn(model,edge_length,3,43,'gtsrb-german-traffic-sign',True)
#%%

#tsr_lib.function('ann',[4,8,12],43,1,'gtsrb-german-traffic-sign')
tsr_lib.function('lr',[5,10,15],43,1,'gtsrb-german-traffic-sign')
#%%

import cv2

image=cv2.imread('btsc/Train/0/01153_00000.ppm')
cv2.imwrite('orig.jpg',image)
cv2.imwrite('blurry.jpg',cv2.blur(image,(5,5)))
cv2.imwrite('r180.jpg',cv2.rotate(image, cv2.ROTATE_180))
cv2.imwrite('r901.jpg',cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
cv2.imwrite('r902.jpg',cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))

#%%
import cv2
#%%
image=cv2.imread('GTSRB/Train/0/00000_00000.ppm')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image=image.reshape(image.shape[0],image.shape[1],1)
#image=tsr_lib.preprocess_img(image,32)
cv2.imshow('xx',image)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
import numpy as np
x=np.array([1,2,3])
y=np.array([1,2,3])
#%%

A=[x,y]

#%%

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%

model1.save('my_model.h5')
del model1
from keras.models import load_model
model1 = load_model('my_model.h5')
model1.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test))

score = model1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])