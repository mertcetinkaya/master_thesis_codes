import numpy as np
from os import listdir,makedirs
from os.path import isfile, join, exists
import cv2
from skimage import transform
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,fbeta_score
import seaborn as sns
import general_lib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from keras.callbacks import ModelCheckpoint

#hog feature data production funciton

def produce_data(ppcs,edge_length,medical_image_directory,total_class_number,training_or_test):
    TOTAL_CLASS_NUMBER=total_class_number
    for ppc in ppcs:
        for class_number in range(0,TOTAL_CLASS_NUMBER):
            files = [f for f in listdir(medical_image_directory+'/binary_images/'+training_or_test+'_all/%d'%class_number) if isfile(join(medical_image_directory+'/binary_images/'+training_or_test+'_all/%d'%class_number, f))]
            image_number=0
            for i in files:
                #path operation
                image_path = medical_image_directory+'/binary_images/'+training_or_test+'_all/%d'%class_number+'/%s'%i
                #hog feature extraction
                if(class_number==0 and image_number==0):
                    hogFeature=general_lib.get_hog_feature(image_path,ppc,edge_length)
                    hogFeature=np.concatenate((hogFeature,class_number),axis=None)
                    hogFeature=hogFeature.reshape((1,hogFeature.shape[0]))
                else:
                    hogFeature2=general_lib.get_hog_feature(image_path,ppc,edge_length)
                    hogFeature2=np.concatenate((hogFeature2,class_number),axis=None)
                    hogFeature2=hogFeature2.reshape((1,hogFeature2.shape[0]))
                    hogFeature=np.concatenate((hogFeature,hogFeature2),axis=0)
                image_number+=1
                print(image_number)
        #path operation
        if(training_or_test=='training'):
            folder_path = medical_image_directory+'/hog/ppcpython%d'%ppc
        elif(training_or_test=='test'):
            folder_path = medical_image_directory+'/hog_test/ppcpython%d'%ppc
        if not exists(folder_path):
            makedirs(folder_path)
        if(training_or_test=='training'):
            saving_path=folder_path+'/data_%d'%ppc+'.npy'
        elif(training_or_test=='test'):
            saving_path=folder_path+'/test_data_%d'%ppc+'.npy'
        
        
        
        if(training_or_test=='training'):
            labels=hogFeature[:,hogFeature.shape[1]-1]
            indices0 = [i for i, x in enumerate(labels) if x == 0]
            indices1 = [i for i, x in enumerate(labels) if x == 1]
            from random import sample
            indices0=sample(indices0,len(indices1))
            indices=indices0+indices1
            new_data=[]
            for i in indices:
                new_data.append(hogFeature[i])
            hogFeature=np.array(new_data)
      
        #saving
        np.save(saving_path,hogFeature)


#hog feature training and test function      
def function(method_name,ppcs,total_class_number,alpha_or_c,medical_image_directory):
    if(alpha_or_c==None):
        alpha_or_c=1
    
    training_acc_list=[]
    test_acc_list=[]
    training_acc_mean_list=[]
    test_acc_mean_list=[]
    training_time_list=[]
    prediction_time_list=[]
    print(method_name.upper())
    print('ALPHA OR C: ',alpha_or_c)
    for ppc in ppcs:
        print('PPC {}'.format(ppc))
        training=np.load(medical_image_directory+'/hog/ppcpython%d'%ppc+'/data_%d.npy'%ppc)
        test=np.load(medical_image_directory+'/hog_test/ppcpython%d'%ppc+'/test_data_%d.npy'%ppc)
        x_training=training[:,0:training.shape[1]-1]
        y_training=training[:,training.shape[1]-1]
        x_test=test[:,0:training.shape[1]-1]
        y_test=test[:,training.shape[1]-1]
        
        if (method_name=='ann'):
            y_training=y_training.reshape((-1,1))
        
                
        training_labels=training[:,training.shape[1]-1]
        training_labels_number=[]
        for i in range(0,total_class_number):
            training_labels_number.append((training_labels==i).sum())
        test_labels=test[:,test.shape[1]-1]
        test_labels_number=[]
        for i in range(0,total_class_number):
            test_labels_number.append((test_labels==i).sum())
        
        
        millis1 = int(round(time.time() * 1000))
        
        if(method_name=='lr'):
            clf = LogisticRegression(C=alpha_or_c).fit(x_training, y_training)
        elif(method_name=='nb'):
            #clf= MultinomialNB(alpha=alpha_or_c).fit(x_training, y_training)
            clf= GaussianNB().fit(x_training, y_training)
        elif(method_name=='svm'):
            clf= LinearSVC(C=alpha_or_c).fit(x_training, y_training)
        elif(method_name=='ann'):
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder()
            y_training_ohe = ohe.fit_transform(y_training).toarray()
            #Dependencies
            # Neural network
            model = Sequential()
            model.add(Dense(128, input_dim=x_training.shape[1], activation='relu'))
            model.add(Dense(total_class_number, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(x_training, y_training_ohe, epochs=25, batch_size=32)
        
        
        millis2 = int(round(time.time() * 1000))
        print('Training time in second: ',(millis2-millis1)/1000)
        training_time_list.append((millis2-millis1)/1000)
        
        millis1 = int(round(time.time() * 1000))
        
        if(method_name=='ann'):
            y_pred = model.predict(x_training)
            training_pred = list()
            for i in range(len(y_pred)):
                training_pred.append(np.argmax(y_pred[i]))
            training_real = list()
            for i in range(len(y_training)):
                training_real.append(int(y_training[i]))
            
        
            y_pred = model.predict(x_test)
            test_pred = list()
            for i in range(len(y_pred)):
                test_pred.append(np.argmax(y_pred[i]))
            test_real = list()
            for i in range(len(y_test)):
                test_real.append(int(y_test[i]))

        else:
            training_pred=clf.predict(x_training)
            training_real=y_training
            test_pred=clf.predict(x_test)
            test_real=y_test
            
        millis2 = int(round(time.time() * 1000))
        print('Prediction time in second: ',(millis2-millis1)/1000)
        prediction_time_list.append((millis2-millis1)/1000)
        
        

        training_accuracy = accuracy_score(training_pred,training_real)

        training_acc_list.append(training_accuracy)
        test_accuracy = accuracy_score(test_pred,test_real)

        test_acc_list.append(test_accuracy)
        
        training_correct_number=total_class_number*[0]
        training_false_number=total_class_number*[0]
        for i in range(0,total_class_number):
            for j in range(len(training_pred)):
                if(training_real[j]==i and training_pred[j]==training_real[j]):
                    training_correct_number[i]+=1
                elif(training_real[j]==i and training_pred[j]!=training_real[j]):
                    training_false_number[i]+=1
        training_correct_rate=[]
        training_total=[]
        for i in range(0,total_class_number):
            training_total.append(training_correct_number[i]+training_false_number[i])
            training_correct_rate.append(training_correct_number[i]/(training_correct_number[i]+training_false_number[i]))
        
        
        test_correct_number=total_class_number*[0]
        test_false_number=total_class_number*[0]
        for i in range(0,total_class_number):
            for j in range(len(test_pred)):
                if(test_real[j]==i and test_pred[j]==test_real[j]):
                    test_correct_number[i]+=1
                elif(test_real[j]==i and test_pred[j]!=test_real[j]):
                    test_false_number[i]+=1
        test_correct_rate=[]
        test_total=[]
        for i in range(0,total_class_number):
            test_total.append(test_correct_number[i]+test_false_number[i])
            if((test_correct_number[i]+test_false_number[i])>0):
                test_correct_rate.append(test_correct_number[i]/(test_correct_number[i]+test_false_number[i]))
            else:
                test_correct_rate.append(0)
            
        training_correct_number=np.array(training_correct_number)
        training_correct_number=training_correct_number.reshape((-1,1))
        training_false_number=np.array(training_false_number)
        training_false_number=training_false_number.reshape((-1,1))
        training_total=np.array(training_total)
        training_total=training_total.reshape((-1,1))
        training_correct_rate=np.array(training_correct_rate)
        training_correct_rate=training_correct_rate.reshape((-1,1))
        
        test_correct_number=np.array(test_correct_number)
        test_correct_number=test_correct_number.reshape((-1,1))
        test_false_number=np.array(test_false_number)
        test_false_number=test_false_number.reshape((-1,1))
        test_total=np.array(test_total)
        test_total=test_total.reshape((-1,1))
        test_correct_rate=np.array(test_correct_rate)
        test_correct_rate=test_correct_rate.reshape((-1,1))
        print('accuracy by class')
        df = pd.DataFrame(data=np.concatenate([training_correct_number,training_false_number,training_total,
                                               training_correct_rate,test_correct_number,test_false_number,
                                               test_total,test_correct_rate],axis=1), 
                          index=range(0,total_class_number), columns=["training_correct", "training_false", "training_total",
                                                      "training_correct_rate","test_correct",
                                                      "test_false", "test_total",
                                                      "test_correct_rate"])
        
        training_acc_mean_list.append(df['training_correct_rate'].mean())
        test_acc_mean_list.append(df['test_correct_rate'].mean())
        print(df)
        
        
        print('PPC {}'.format(ppc))
        print('Training accuracy {}'.format(training_accuracy))
        print('Test accuracy {}'.format(test_accuracy))
        print('Training accuracy mean {}'.format(df['training_correct_rate'].mean()))
        print('Test accuracy mean {}'.format(df['test_correct_rate'].mean()))
        
        
        x=range(0,total_class_number)
        y1=df['training_correct_rate']
        y2=df['test_correct_rate']
        fig = plt.figure(figsize=(20, 8))
        fig.add_subplot(121)
        plt.plot(x,y1,'ko-')
        plt.plot(x,y2,'ro-')
        plt.xticks(np.arange(min(x), max(x), 5))
        plt.xlabel('class')
        plt.ylabel('accuracy')
        plt.title('accuracy by class')
        plt.legend(['Training', 'Test'], loc='upper right')
        #plt.show()
        fig.add_subplot(122)
        data = [training_labels_number,test_labels_number]
        X = np.arange(0,total_class_number)
        #fig = plt.figure()
        #ax2.add_axes([0,0,1,1])
        plt.bar(X + 0.00, data[0], color = 'k', width = 0.25)
        plt.bar(X + 0.25, data[1], color = 'r', width = 0.25)
        plt.legend(labels=['Training', 'Test'])
        plt.xticks(np.arange(0, total_class_number, 5))
        plt.yticks(np.arange(0, max(training_labels_number), 250))
        plt.xlabel('class')
        plt.ylabel('frequency')
        plt.title('class number distribution plot')
        plt.show()
        
    
    
    fig=plt.figure(figsize=(20, 8))
    fig.add_subplot(121)
    plt.plot(ppcs,training_acc_list,'ko-')
    plt.plot(ppcs,test_acc_list,'ro-')
    plt.plot(ppcs,training_acc_mean_list,'k^-')
    plt.plot(ppcs,test_acc_mean_list,'r^-')
    
    plt.xlabel('ppc')
    plt.ylabel('accuracy')
    plt.title('overall accuracy by ppc')
    plt.legend(['Training_accuracy', 'Test_accuracy','Training_accuracy_mean', 'Test_accuracy_mean'], 
               loc='upper right' , prop={'size': 8}) 
    fig.add_subplot(122)
    plt.plot(ppcs,training_time_list,'ko-')
    plt.plot(ppcs,prediction_time_list,'ro-')
    plt.xlabel('ppc')
    plt.ylabel('time')
    plt.title('training and prediction time in second for each ppc')
    plt.legend(['Training', 'Prediction'], loc='upper right')
    plt.show()

#image data reading function
def read_data(medical_image_directory,downsampling=True):
    data=[]
    labels=[]

    for i in range(2) :
        path=medical_image_directory+'/binary_images/training_all/{}/'.format(i)
        Class=os.listdir(path)
        for a in Class:
            image=cv2.imread(path+a)
            data.append(np.array(image))
            labels.append(i)
            
    x_training=np.array(data)
    labels=np.array(labels)


    y_training=labels
    
    validation_size=0.25
    if(validation_size<1):
        x_training, x_val, y_training, y_val= train_test_split(x_training,y_training,stratify=y_training,test_size=validation_size,random_state=42)
        
    if(downsampling==True):      
        indices0 = [i for i, x in enumerate(y_training) if x == 0]
        indices1 = [i for i, x in enumerate(y_training) if x == 1]
        
        if(len(indices0) > len(indices1)):
            from random import sample
            indices0=sample(indices0,len(indices1))
            indices=indices0+indices1
            new_x_training=[]
            for i in indices:
                new_x_training.append(x_training[i])
            new_y_training=[]
            for i in indices:
                new_y_training.append(y_training[i])
            x_training=np.array(new_x_training)
            y_training=np.array(new_y_training)




    #Using one hote encoding for the train and validation labels
    y_training = to_categorical(y_training, 2)
    y_val = to_categorical(y_val, 2)
        
    y_test=pd.read_csv(medical_image_directory+"/Test_All.csv")
    #labels=y_test['Path'].as_matrix()
    labels=y_test['Path'].to_numpy()
    y_test=y_test['ClassId'].values
    

    data=[]

    for f in labels:
        image=cv2.imread(medical_image_directory+'/'+f)
        data.append(np.array(image))

    x_test=np.array(data)

    return (x_training,x_val,x_test,y_training,y_val,y_test)

#data production function for cnn
def cnn_produce_data(medical_image_directory,edge_length,downsampling=True):
    data=[]
    labels=[]


    for i in range(2) :
        path=medical_image_directory+'/binary_images/training_all/{}/'.format(i)
        Class=os.listdir(path)
        for a in Class:
            image=cv2.imread(path+a)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image=transform.resize(image, (edge_length, edge_length))
            data.append(np.array(image))
            labels.append(i)
            

    x_training=np.array(data)
    labels=np.array(labels)
    y_training=labels
    
      
    
    validation_size=0.25
    if(validation_size<1):
        x_training, x_val, y_training, y_val= train_test_split(x_training,y_training,stratify=y_training,test_size=validation_size,random_state=42)
        
    if(downsampling==True):      
        indices0 = [i for i, x in enumerate(y_training) if x == 0]
        indices1 = [i for i, x in enumerate(y_training) if x == 1]
        
        if(len(indices0) > len(indices1)):
            from random import sample
            indices0=sample(indices0,len(indices1))
            indices=indices0+indices1
            new_x_training=[]
            for i in indices:
                new_x_training.append(x_training[i])
            new_y_training=[]
            for i in indices:
                new_y_training.append(y_training[i])
            x_training=np.array(new_x_training)
            y_training=np.array(new_y_training)
            

    #Using one hote encoding for the train and validation labels
    y_training = to_categorical(y_training, 2)
    y_val = to_categorical(y_val, 2)
        
    y_test=pd.read_csv(medical_image_directory+"/Test_All.csv")
    #labels=y_test['Path'].as_matrix()
    labels=y_test['Path'].to_numpy()
    y_test=y_test['ClassId'].values
    

    data=[]

    for f in labels:
        image=cv2.imread(medical_image_directory+'/'+f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=transform.resize(image, (edge_length, edge_length))
        data.append(np.array(image))

    x_test=np.array(data)

    return (x_training,x_val,x_test,y_training,y_val,y_test)


#data production with data augmentation function for cnn 
def cnn_produce_data_augmentation(medical_image_directory,edge_length,downsampling=True):
    data=[]
    labels=[]


    for i in range(2) :
        path=medical_image_directory+'/binary_images/training_all/{}/'.format(i)
        Class=os.listdir(path)
        for a in Class:
            image=cv2.imread(path+a)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data.append(np.array(image))
            labels.append(i)
            

    x_training=np.array(data)
    labels=np.array(labels)
    y_training=labels
    
    validation_size=0.25
    if(validation_size<1):
        x_training, x_val, y_training, y_val= train_test_split(x_training,y_training,stratify=y_training,test_size=validation_size,random_state=42)
        
    if(downsampling==True):      
        indices0 = [i for i, x in enumerate(y_training) if x == 0]
        indices1 = [i for i, x in enumerate(y_training) if x == 1]
        
        if(len(indices0) > len(indices1)):
            from random import sample
            indices0=sample(indices0,len(indices1))
            indices=indices0+indices1
            new_x_training=[]
            for i in indices:
                new_x_training.append(x_training[i])
            new_y_training=[]
            for i in indices:
                new_y_training.append(y_training[i])
            x_training=np.array(new_x_training)
            y_training=np.array(new_y_training)
            
    
    orig_x_training_length=len(x_training)
    x_training=list(x_training)
    y_training=list(y_training)
    for i in range(orig_x_training_length):
        image=x_training[i].copy()
        x_training[i]=transform.resize(image, (edge_length, edge_length))
        x_training.append(transform.resize(cv2.blur(image,(5,5)), (edge_length, edge_length)))
        y_training.append(y_training[i])
        x_training.append(transform.resize(cv2.rotate(image,cv2.ROTATE_180), (edge_length, edge_length)))
        y_training.append(y_training[i])
        x_training.append(transform.resize(cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE), (edge_length, edge_length)))
        y_training.append(y_training[i])
        x_training.append(transform.resize(cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE), (edge_length, edge_length)))
        y_training.append(y_training[i])
    x_training=np.array(x_training)
    y_training=np.array(y_training)
    
    
    orig_x_val_length=len(x_val)
    x_val=list(x_val)
    for i in range(orig_x_val_length):
        image=x_val[i].copy()
        x_val[i]=transform.resize(image, (edge_length, edge_length))
    x_val=np.array(x_val)

    #Using one hote encoding for the train and validation labels
    y_training = to_categorical(y_training, 2)
    y_val = to_categorical(y_val, 2)
        
    y_test=pd.read_csv(medical_image_directory+"/Test_All.csv")
    #labels=y_test['Path'].as_matrix()
    labels=y_test['Path'].to_numpy()
    y_test=y_test['ClassId'].values
    

    data=[]

    for f in labels:
        image=cv2.imread(medical_image_directory+'/'+f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=transform.resize(image, (edge_length, edge_length))
        data.append(np.array(image))

    x_test=np.array(data)
    return (x_training,x_val,x_test,y_training,y_val,y_test)

#cnn probability threshold moving function
def cnn_probability_moving(model,training_val_test_data,medical_image_directory):
    x_training,x_val,x_test,y_training,y_val,y_test=training_val_test_data[0],training_val_test_data[1],training_val_test_data[2],training_val_test_data[3],training_val_test_data[4],training_val_test_data[5]
    y_training=np.argmax(y_training,axis=1)
    y_val=np.argmax(y_val,axis=1)
    y_training_hat = model.predict_proba(x_training)
    y_training_hat = y_training_hat[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_training, y_training_hat)
    beta=2
    fbeta = ((1+beta*beta) * precision * recall) / ((beta*beta)*precision + recall)
    ix = np.argmax(fbeta)
    thresh=thresholds[ix]
    print('The Best Threshold= ', thresh)
    plt.plot(recall, precision, marker='.', label='results',zorder=1)
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best',zorder=2)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.savefig(medical_image_directory+'/pr_re_curve.jpg')
    plt.show()

    print('Training precision: ',precision_score(y_training,np.where(model.predict_proba(x_training)[:,1] > thresh,1,0)))
    print('Training recall: ',recall_score(y_training,np.where(model.predict_proba(x_training)[:,1] > thresh,1,0)))
    print('Training fbeta: ',fbeta_score(y_training,np.where(model.predict_proba(x_training)[:,1] > thresh,1,0),beta=2))
    print('Training accuracy: ',accuracy_score(y_training,np.where(model.predict_proba(x_training)[:,1] > thresh,1,0)))
    
    print('Val precision: ',precision_score(y_val,np.where(model.predict_proba(x_val)[:,1] > thresh,1,0)))
    print('Val recall: ',recall_score(y_val,np.where(model.predict_proba(x_val)[:,1] > thresh,1,0)))
    print('Val fbeta: ',fbeta_score(y_val,np.where(model.predict_proba(x_val)[:,1] > thresh,1,0),beta=2))
    print('Val accuracy: ',accuracy_score(y_val,np.where(model.predict_proba(x_val)[:,1] > thresh,1,0)))
    
    print('Test precision: ',precision_score(y_test,np.where(model.predict_proba(x_test)[:,1] > thresh,1,0)))
    print('Test recall: ',recall_score(y_test,np.where(model.predict_proba(x_test)[:,1] > thresh,1,0)))
    print('Test fbeta: ',fbeta_score(y_test,np.where(model.predict_proba(x_test)[:,1] > thresh,1,0),beta=2))
    print('Test accuracy: ',accuracy_score(y_test,np.where(model.predict_proba(x_test)[:,1] > thresh,1,0)))
    
    

#training and test function for cnn
def cnn(model,edge_length,epochs,medical_image_directory,model_number,class_weight,health_threshold,training_val_test_data=None,checkpoint=False):
    
    total_class_number = 2
    print('CNN')
    if(training_val_test_data== None):
        x_training,x_val,x_test,y_training,y_val,y_test=cnn_produce_data(medical_image_directory,edge_length)
    else:
        x_training,x_val,x_test,y_training,y_val,y_test=training_val_test_data[0],training_val_test_data[1],training_val_test_data[2],training_val_test_data[3],training_val_test_data[4],training_val_test_data[5]

   
    
    millis1 = int(round(time.time() * 1000))
    if(checkpoint==True):
        #filepath="best_model_"+str(model_number)+".h5"
        #filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        filepath="best_model_weights_"+str(model_number)+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = model.fit(x_training, y_training, batch_size=32, epochs=epochs,validation_data=(x_val, y_val),callbacks=callbacks_list)
        #history = model.fit(x_training, y_training, batch_size=32, epochs=epochs,validation_data=(x_training, y_training),callbacks=callbacks_list)
    else:
        history = model.fit(x_training, y_training, batch_size=32, epochs=epochs,validation_data=(x_val, y_val))
        #history = model.fit(x_training, y_training, batch_size=32, epochs=epochs,validation_data=(x_training, y_training),callbacks=callbacks_list)
        #model = load_model(filepath)
    millis2 = int(round(time.time() * 1000))
    print('Training time in second: ',(millis2-millis1)/1000)
    
    plt.plot(history.history['accuracy'], label='training accuracy',color='black')
    plt.plot(history.history['val_accuracy'], label='val accuracy',color='blue')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(medical_image_directory+'/history'+str(model_number)+'.jpg')
    plt.show()



    y_training_decoded=[]
    y_val_decoded=[]

    for i in range(len(y_training)):
        y_training_decoded.append(np.argmax(y_training[i]))
    for i in range(len(y_val)):
        y_val_decoded.append(np.argmax(y_val[i]))
    
    y_training_decoded=np.array(y_training_decoded)
    y_val_decoded=np.array(y_val_decoded)
    millis1 = int(round(time.time() * 1000))
    
    y_training_predicted=model.predict_classes(x_training)
    y_val_predicted=model.predict_classes(x_val)
    y_test_predicted=model.predict_classes(x_test)

    
    
    
    millis2 = int(round(time.time() * 1000))
    print('Prediction time in second: ',(millis2-millis1)/1000)



    training_pred=y_training_predicted
    training_real=y_training_decoded
    val_pred=y_val_predicted
    val_real=y_val_decoded
    test_pred=y_test_predicted
    test_real=y_test




    training_correct_number=total_class_number*[0]
    training_false_number=total_class_number*[0]
    for i in range(0,total_class_number):
        for j in range(len(training_pred)):
            if(training_real[j]==i and training_pred[j]==training_real[j]):
                training_correct_number[i]+=1
            elif(training_real[j]==i and training_pred[j]!=training_real[j]):
                training_false_number[i]+=1
    training_correct_rate=[]
    training_total=[]
    for i in range(0,total_class_number):
        training_total.append(training_correct_number[i]+training_false_number[i])
        training_correct_rate.append(training_correct_number[i]/(training_correct_number[i]+training_false_number[i]))

    val_correct_number=total_class_number*[0]
    val_false_number=total_class_number*[0]
    for i in range(0,total_class_number):
        for j in range(len(val_pred)):
            if(val_real[j]==i and val_pred[j]==val_real[j]):
                val_correct_number[i]+=1
            elif(val_real[j]==i and val_pred[j]!=val_real[j]):
                val_false_number[i]+=1
    val_correct_rate=[]
    val_total=[]
    for i in range(0,total_class_number):
        val_total.append(val_correct_number[i]+val_false_number[i])
        val_correct_rate.append(val_correct_number[i]/(val_correct_number[i]+val_false_number[i]))

    
    test_correct_number=total_class_number*[0]
    test_false_number=total_class_number*[0]
    for i in range(0,total_class_number):
        for j in range(len(test_pred)):
            if(test_real[j]==i and test_pred[j]==test_real[j]):
                test_correct_number[i]+=1
            elif(test_real[j]==i and test_pred[j]!=test_real[j]):
                test_false_number[i]+=1
    test_correct_rate=[]
    test_total=[]
    for i in range(0,total_class_number):
        test_total.append(test_correct_number[i]+test_false_number[i])
        if((test_correct_number[i]+test_false_number[i])>0):
            test_correct_rate.append(test_correct_number[i]/(test_correct_number[i]+test_false_number[i]))
        else:
            test_correct_rate.append(0)
        
    training_correct_number=np.array(training_correct_number)
    training_correct_number=training_correct_number.reshape((-1,1))
    training_false_number=np.array(training_false_number)
    training_false_number=training_false_number.reshape((-1,1))
    training_total=np.array(training_total)
    training_total=training_total.reshape((-1,1))
    training_correct_rate=np.array(training_correct_rate)
    training_correct_rate=training_correct_rate.reshape((-1,1))

    val_correct_number=np.array(val_correct_number)
    val_correct_number=val_correct_number.reshape((-1,1))
    val_false_number=np.array(val_false_number)
    val_false_number=val_false_number.reshape((-1,1))
    val_total=np.array(val_total)
    val_total=val_total.reshape((-1,1))
    val_correct_rate=np.array(val_correct_rate)
    val_correct_rate=val_correct_rate.reshape((-1,1))
    
    
    
    test_correct_number=np.array(test_correct_number)
    test_correct_number=test_correct_number.reshape((-1,1))
    test_false_number=np.array(test_false_number)
    test_false_number=test_false_number.reshape((-1,1))
    test_total=np.array(test_total)
    test_total=test_total.reshape((-1,1))
    test_correct_rate=np.array(test_correct_rate)
    test_correct_rate=test_correct_rate.reshape((-1,1))

    print('accuracy by class')
    df = pd.DataFrame(data=np.concatenate([training_correct_number,training_false_number,training_total,
                                           training_correct_rate,val_correct_number,val_false_number,val_total,
                                           val_correct_rate,test_correct_number,test_false_number,
                                           test_total,test_correct_rate],axis=1), 
                      index=range(0,total_class_number), columns=["training_correct", "training_false", "training_total",
                                                  "training_correct_rate","val_correct", "val_false", "val_total",
                                                  "val_correct_rate","test_correct",
                                                  "test_false", "test_total",
                                                  "test_correct_rate"])


    print(df)


    
    print('Training accuracy {}'.format(accuracy_score(y_training_decoded, y_training_predicted)))
    print('Val accuracy {}'.format(accuracy_score(y_val_decoded, y_val_predicted)))
    print('Test accuracy {}'.format(accuracy_score(y_test, y_test_predicted)))
    print('Training accuracy mean ',df['training_correct_rate'].mean())
    print('Val accuracy mean ',df['val_correct_rate'].mean())
    print('Test accuracy mean ',df['test_correct_rate'].mean())
    print('Training precision {}'.format(precision_score(y_pred=training_pred,y_true=training_real)))
    print('Val precision {}'.format(precision_score(y_pred=val_pred,y_true=val_real)))
    print('Test precision {}'.format(precision_score(y_pred=test_pred,y_true=test_real)))
    print('Training recall {}'.format(recall_score(y_pred=training_pred,y_true=training_real)))
    print('Val recall {}'.format(recall_score(y_pred=val_pred,y_true=val_real)))
    print('Test recall {}'.format(recall_score(y_pred=test_pred,y_true=test_real)))
    print('Training f1-measure {}'.format(f1_score(y_pred=training_pred,y_true=training_real)))
    print('Val f1-measure {}'.format(f1_score(y_pred=val_pred,y_true=val_real)))
    print('Test f1-measure {}'.format(f1_score(y_pred=test_pred,y_true=test_real)))
    
    
    fig = plt.figure(figsize=(10, 8))
    #fig.add_subplot(121)
    sns.heatmap(confusion_matrix(y_pred=training_pred,y_true=training_real), annot=True,fmt='g')
    plt.title('training confusion matrix')
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.savefig(medical_image_directory+'/tr_conf'+str(model_number)+'.jpg')
    plt.show()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_pred=y_val_predicted,y_true=y_val_decoded,labels=range(total_class_number)),  annot=True,fmt='g',annot_kws={"size":8})
    plt.title('val confusion matrix')
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.savefig(medical_image_directory+'/val_conf'+str(model_number)+'.jpg')
    plt.show()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_pred=test_pred,y_true=test_real), annot=True,fmt='g')
    plt.title('test confusion matrix')
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.savefig(medical_image_directory+'/test_conf'+str(model_number)+'.jpg')
    plt.show()
    

    training_labels=y_training_decoded
    training_labels_number=[]
    for i in range(0,total_class_number):
        training_labels_number.append((training_labels==i).sum())
    val_labels=y_val_decoded
    val_labels_number=[]
    for i in range(0,total_class_number):
        val_labels_number.append((val_labels==i).sum())
    test_labels=y_test
    test_labels_number=[]
    for i in range(0,total_class_number):
        test_labels_number.append((test_labels==i).sum())



    x=range(0,total_class_number)
    y1=df['training_correct_rate']
    y2=df['val_correct_rate']
    y3=df['test_correct_rate']
#    fig = plt.figure(figsize=(20, 8))
#    fig.add_subplot(121)
#    plt.plot(x,y1,'ko-')
#    plt.plot(x,y2,'bo-',alpha=0.5)
#    plt.plot(x,y3,'ro-',alpha=0.5)
#    plt.xticks(np.arange(min(x), total_class_number))
#    plt.xlabel('class')
#    plt.ylabel('accuracy')
#    plt.title('accuracy by class')
#    plt.legend(['Training', 'Val','Test'], loc='upper right')


    #fig.add_subplot(122)
    data = [training_labels_number,val_labels_number,test_labels_number]
    X = np.arange(0,total_class_number)
    plt.bar(X + 0.00, data[0], color = 'k', width = 0.25)
    plt.bar(X + 0.25, data[1], color = 'b', width = 0.25)
    plt.bar(X + 0.5, data[2], color = 'r', width = 0.25)
    plt.legend(labels=['Training', 'Val','Test'])
    plt.xticks(np.arange(0, total_class_number))
    plt.xlabel('class')
    plt.ylabel('frequency')
    plt.title('class number distribution plot')
    plt.savefig(medical_image_directory+'/general'+str(model_number)+'.jpg')
    plt.show()
    
    
    return accuracy_score(y_training_decoded, y_training_predicted), accuracy_score(y_val_decoded, y_val_predicted), accuracy_score(y_test, y_test_predicted), model
    
   
