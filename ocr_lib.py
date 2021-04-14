import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from skimage import exposure, feature, transform
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
random.seed(0)
#%%
boundaries=[]
width_height=[]
def preprocessing(img,binary_thresh):
    lower_boundary=0
    original_length=img.shape[0]
    upper_boundary=img.shape[0]
    if(len(img.shape)==3 and img.shape[2]==3):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,binary_thresh,255,cv2.THRESH_BINARY)[1]
    else:
        thresh=img
    
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if(thresh[i,j]==0):
                upper_boundary=i+1
                #upper_boundary=i+int((thresh.shape[0]-i)/2)
                break
        
    
    for i in range(thresh.shape[0]-1,0,-1):
        for j in range(thresh.shape[1]):
            if(thresh[i,j]==0):
                lower_boundary=i-1
                #lower_boundary=int(i/2)
                break
        
    
    
    if(len(img.shape)==3):
        img=img[lower_boundary:upper_boundary,:,:]
        width=img.shape[1]
    else:
        img=img[lower_boundary:upper_boundary,:]
        width=img.shape[1]
    return img,lower_boundary,upper_boundary,original_length,width

def find_borders(thresh):
    #finding horizontal borders with a trick for characters like i,ü,ğ,İ,Ü,Ğ with extension at the top
    horizontal_borders=[]
    in_the_line=0
    for i in range(thresh.shape[0]):
        if((in_the_line==0) and np.any(thresh[i][:]!=255)):
            in_the_line=1
            horizontal_borders.append(i-1)
        elif((in_the_line==1) and np.all(thresh[i][:]==255)):
            in_the_line=0
            horizontal_borders.append(i)
        #only below is for the characters with extension at the top  
        x=len(horizontal_borders)
        if(x>=4 and (horizontal_borders[x-2]-horizontal_borders[x-4])<0.5*(horizontal_borders[x-1]-horizontal_borders[x-2])):
            del horizontal_borders[(x-3):(x-1)]


    #finding vertical borders and rectangle borders        
    vertical_borders=[]
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    in_the_line=0
    for j in range(0,len(horizontal_borders),2):
        for i in range(thresh.shape[1]):
            if(in_the_line==0):
                for k in range(horizontal_borders[j],horizontal_borders[j+1]):
                    if(thresh[k][i]!=255):
                        in_the_line=1
                        vertical_borders.append(i-1)
                        x1.append(i-1)
                        y1.append(horizontal_borders[j])
                        break
            if(in_the_line==1):
                counter=0
                for k in range(horizontal_borders[j],horizontal_borders[j+1]):
                    if(thresh[k][i]==255):
                        if(k<(horizontal_borders[j+1]-1)):
                            continue
                        else:
                            in_the_line=0
                            vertical_borders.append(i)
                            x2.append(i)
                            y2.append(horizontal_borders[j+1])
                    else:
                        counter+=1
                        if(counter>0):
                            break
    return x1,y1,x2,y2


def produce_training_data(training_image_list,alphabet,binary_thresh):
    number=0
    for training_image_number in range(len(training_image_list)):
        training_image=training_image_list[training_image_number]
        gray = cv2.cvtColor(training_image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,binary_thresh,255,cv2.THRESH_BINARY)[1]
        x1,y1,x2,y2=find_borders(thresh)
        for i in range(len(x1)):
            repeat=1
            for j in range(repeat):
                im=training_image[y1[i]:y2[i],x1[i]:x2[i],0:3]
                im,x,y,z,t=preprocessing(im,binary_thresh)
                boundaries.append([x,y,z])
                width_height.append([t,y-x])
                cv2.imwrite('ocr_training/images/%d.jpg'%number,im)
                number+=1
   

                
def extra_correction(predictions,probabilities,boundaries,alphabet):
    for i in range(len(predictions)):
        if(alphabet[predictions[i]] in ['.',','] and boundaries[i][0]<=0.1*boundaries[i][2]):   
            predictions[i]=alphabet.index('`')
        elif(alphabet[predictions[i]]=='`' and boundaries[i][0]>0.1*boundaries[i][2]):
            x=np.argmax(np.array([probabilities[i,alphabet.index('.')],probabilities[i,alphabet.index(',')]]))
            if(x==0): predictions[i]=alphabet.index('.')
            elif(x==1): predictions[i]=alphabet.index(',')
        elif(alphabet[predictions[i]]=='-' and boundaries[i][1]>0.7*boundaries[i][2]):
            x=np.argmax(np.array([probabilities[i,alphabet.index('.')],probabilities[i,alphabet.index(',')]]))
            if(x==0): predictions[i]=alphabet.index('.')
            elif(x==1): predictions[i]=alphabet.index(',')

        elif(alphabet[predictions[i]] in ['ı','I','i','İ','l','1']):
            if(boundaries[i][0]>0.5*boundaries[i][2]):
                x=np.argmax(np.array([probabilities[i,alphabet.index('.')],probabilities[i,alphabet.index(',')]]))
                if(x==0): predictions[i]=alphabet.index('.')
                elif(x==1): predictions[i]=alphabet.index(',')
            elif((boundaries[i][1]-boundaries[i][0])<=0.35*boundaries[i][2]):
                predictions[i]=alphabet.index('`')
            elif(0.35*(boundaries[i][2])<(boundaries[i][1]-boundaries[i][0])<=0.6*(boundaries[i][2])):
                predictions[i]=alphabet.index('ı')
            elif(0.6*(boundaries[i][2])<(boundaries[i][1]-boundaries[i][0])):
                x=np.argmax(np.array([probabilities[i,alphabet.index('I')],probabilities[i,alphabet.index('i')],
                                    probabilities[i,alphabet.index('İ')],probabilities[i,alphabet.index('l')],
                                    probabilities[i,alphabet.index('1')]]))
                if(x==0): predictions[i]=alphabet.index('I')
                elif(x==1): predictions[i]=alphabet.index('i')
                elif(x==2): predictions[i]=alphabet.index('İ')
                elif(x==3): predictions[i]=alphabet.index('l')
                elif(x==4): predictions[i]=alphabet.index('1')

        elif(alphabet[predictions[i]] in ['C','Ç','O','P','S','Ş','U','V','W','X','Y','Z'] and boundaries[i][0]>=0.175*boundaries[i][2]):
            predictions[i]=predictions[i]+1
        elif(alphabet[predictions[i]] in ['c','ç','o','p','s','ş','u','v','w','x','y','z'] and boundaries[i][0]<=0.175*boundaries[i][2]):
            predictions[i]=predictions[i]-1
        #elif(alphabet[predictions[i]] in ['Ö','Ü'] and boundaries[i][0]>0):
            #predictions[i]=predictions[i]+1
        #elif(alphabet[predictions[i]] in ['ö','ü'] and boundaries[i][0]==0):
            #predictions[i]=predictions[i]-1

            
def transform_text_to_alphabet(text):
    alphabet=[]
    for i in text:
        alphabet.append(i)
    return alphabet

def remove_lines(image):
    image1=image.copy()
    gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image1, [c], -1, (255,255,255), 2)
        
        
    # Remove vertical
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    # detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, (255,255,255), 2)
    
    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    result = 255 - cv2.morphologyEx(255 - image1, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    
    return result

def ocr(method,alphabet,test_image,f_binary_thresh,s_binary_thresh,training_image_list,blurring,line_removing=False):
    classes=[]
    for training_image_number in range(len(training_image_list)):
        for i in range(len(alphabet)):
            classes.append(i)
            
    data=[]
    labels=[]

    number=0
    for i in range(len(classes)) :
        path="ocr_training/images/"
        image=cv2.imread(path+'%d.jpg'%number)
        if (method!='cnn'):
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray,f_binary_thresh,255,cv2.THRESH_BINARY)[1]
            image=thresh
        
        #image_from_array = Image.fromarray(image)
        #size_image = image_from_array.resize((30,30))
        size_image=transform.resize(image, (30, 30))
        data.append(np.array(size_image))
        labels.append(classes[i])
        number+=1

    
    x_training=np.array(data)
    #x_training = x_training.astype('float32')/255
    if(method!='cnn'):
        x_training=x_training.reshape((x_training.shape[0],x_training.shape[1]*x_training.shape[2]))
    labels=np.array(labels)


    #x_training = x_training.astype('float32')/255 
    y_training=labels
      
    #x_training, y_training = shuffle(x_training, y_training)
    indices=np.arange(len(x_training))
    x_training,x_test,y_training,y_test,idx_training,idx_test=train_test_split(x_training,y_training,indices,test_size=0.25,random_state=42)
    
    
    boundaries_training=[boundaries[i] for i in idx_training]
    width_height_training=[width_height[i] for i in idx_training]
    boundaries_test=[boundaries[i] for i in idx_test]
    #width_height_test=[width_height[i] for i in idx_test]
    
    
    data1=[item[0] for item in width_height_training]
    data2=[item[1] for item in width_height_training]
    data = pd.DataFrame(list(zip(data1, data2)),columns =['width', 'height'])
    myfig = plt.figure()
    data.boxplot(column=['width', 'height'],grid=False,figsize=(8,8))
    myfig.suptitle('width and height values')
    myfig.savefig("boxplot.jpg")
    
    
    
    df = pd.DataFrame(boundaries_training)
    df.columns=['first', 'second', 'length']
    df['first/second']=df['first']/df['second']
    df['second/length']=df['second']/df['length']
    df['first/length']=df['first']/df['length']
    df['char']=[alphabet[i] for i in y_training]
    df=df.set_index(keys='char')
    
    fig=plt.figure(figsize=(20,10)) 
    plt.subplots_adjust(hspace = 0.4)
    fig.suptitle('ratios between first border, second border and original length of character',y=0.95)
    
    rates=['first/second','second/length','first/length']
    for i in range(1,4):
        plt.subplot(3,1,i)
        plt.title(rates[i-1])
        plt.plot(df.groupby('char').agg(['min'])[rates[i-1]].reindex(transform_text_to_alphabet(alphabet)),'r.-'
                 ,label='min')
        plt.plot(df.groupby('char').agg(['mean'])[rates[i-1]].reindex(transform_text_to_alphabet(alphabet)),'g.-'
                ,label='mean')
        plt.plot(df.groupby('char').agg(['max'])[rates[i-1]].reindex(transform_text_to_alphabet(alphabet)),'b.-'
                ,label='max')
        plt.legend()
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('character')
        plt.ylabel('ratio')
    plt.savefig('extra_correction.jpg')
    plt.show()
    

    
    #Using one hote encoding for the train and validation labels
    if(method=='cnn' or method=='ann'):
        y_training = to_categorical(y_training, len(alphabet))

    millis1 = int(round(time.time() * 1000))

        
    if(method=='cnn'):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',input_shape=(30,30,3)))
        #model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        #model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        #model.add(MaxPool2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(len(alphabet), activation='softmax'))


        #Compilation of the model

        #lr = 0.01
        #sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(x_training, y_training, epochs=50)

        
        
    elif(method=='ann'):
        model = Sequential()
        model.add(Dense(300, input_dim=x_training.shape[1], activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(alphabet), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_training, y_training, epochs=300)

        
    elif(method=='lr'):
        model = LogisticRegression(random_state = 42).fit(x_training, y_training)
        
    elif(method=='knn'):
        #from sklearn.neighbors import NeighborhoodComponentsAnalysis
        #nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
        #nca.fit(x_training, y_training)
        #x_training = nca.transform(x_training)
        model = KNeighborsClassifier().fit(x_training, y_training)
        
    elif(method=='nb'):
        model= GaussianNB().fit(x_training, y_training)

    elif(method=='svm'):
        model= SVC(kernel='linear',probability=True,random_state = 42).fit(x_training, y_training)
        #model= LinearSVC().fit(x_training, y_training)

    elif(method=='rf'):
        model= RandomForestClassifier().fit(x_training, y_training)

        
    millis2 = int(round(time.time() * 1000))
    print('Training time in second: ',(millis2-millis1)/1000)
    
    
    if(method in ['cnn','ann']):
        y_pred=model.predict_classes(x_test)
        y_prob=model.predict_proba(x_test)
        y_pred_training=model.predict_classes(x_training)
        y_prob_training=model.predict_proba(x_training)
    elif(method in ['lr','nb','svm','rf','knn']):
        y_pred=model.predict(x_test)
        y_prob=model.predict_proba(x_test)
        y_pred_training=model.predict(x_training)
        y_prob_training=model.predict_proba(x_training)
    # elif(method=='knn'):
    #     x_test = nca.transform(x_test)
    #     y_pred=model.predict(x_test)
    #     y_prob=model.predict_proba(x_test)
    #     y_pred_training=model.predict(x_training)
    #     y_prob_training=model.predict_proba(x_training)
    
    if(method in ['cnn','ann']):
        print('Training accuracy before extra correction: ',accuracy_score(np.argmax(y_training,axis=1),y_pred_training))
    else:
        print('Training accuracy before extra correction: ',accuracy_score(y_training,y_pred_training))
    print('Test accuracy before extra correction: ',accuracy_score(y_test,y_pred))
        
    plt.figure(figsize=(45,40)) 
    plt.title('Confusion Matrix for Test Data')
    cm=confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.25)
    sns.heatmap(cm,xticklabels=transform_text_to_alphabet(alphabet),yticklabels=transform_text_to_alphabet(alphabet),annot=True)
    
    plt.xticks(rotation=0) 
    plt.yticks(rotation=0) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('cm_before_corr.jpg')
    plt.show()
    extra_correction(predictions=y_pred,probabilities=y_prob,boundaries=boundaries_test,alphabet=alphabet)
    extra_correction(predictions=y_pred_training,probabilities=y_prob_training,boundaries=boundaries_training,alphabet=alphabet)
    
    if(method in ['cnn','ann']):
        print('Training accuracy after extra correction: ',accuracy_score(np.argmax(y_training,axis=1),y_pred_training))
    else:
        print('Training accuracy after extra correction: ',accuracy_score(y_training,y_pred_training))
    print('Test accuracy after extra correction: ',accuracy_score(y_test,y_pred))
    
    plt.figure(figsize=(45,40)) 
    plt.title('Confusion Matrix for Test Data')
    cm=confusion_matrix(y_test, y_pred)
    sns.heatmap(cm,xticklabels=transform_text_to_alphabet(alphabet),yticklabels=transform_text_to_alphabet(alphabet),annot=True)
    plt.xticks(rotation=0) 
    plt.yticks(rotation=0) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('cm_after_corr.jpg')
    plt.show()
    
    
    
    #test
    img_orig=test_image
    if(line_removing==False):
        img=cv2.medianBlur(img_orig,blurring)
        cv2.imshow('blur', cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))))
        cv2.imwrite('blur.jpg', cv2.resize(img,(int(img.shape[1]/1),int(img.shape[0]/1))))
        gray = cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY)
    else:
        img=cv2.medianBlur(remove_lines(img_orig),blurring)
        gray = cv2.cvtColor(remove_lines(img_orig),cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray,s_binary_thresh,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', cv2.resize(thresh,(int(thresh.shape[1]/2),int(thresh.shape[0]/2))))
    cv2.imwrite('thresh.jpg', cv2.resize(thresh,(int(thresh.shape[1]/1),int(thresh.shape[0]/1))))
    thresh = cv2.medianBlur(thresh, blurring)

    x1,y1,x2,y2=find_borders(thresh)

              

    
    #segmentation
    data=[]
    lower_b=[]
    upper_b=[]
    original_l=[]
    boundaries_test=[]
    for i in range(len(x1)):
        if(method=='cnn'):
            #im=img[y1[i]:y2[i],x1[i]:x2[i],0:3]
            im,a,b,c=preprocessing(img[y1[i]:y2[i],x1[i]:x2[i],0:3],s_binary_thresh)[0:4]
            boundaries_test.append([a,b,c])
            lower_b.append(a)
            upper_b.append(b)
            original_l.append(c)
            cv2.imwrite('ocr_character/%d.jpg'%i,im)
        else:
            #cv2.imshow('thresh', cv2.resize(thresh,(int(thresh.shape[1]/2),int(thresh.shape[0]/2))))
            #cv2.imwrite('thresh.jpg', cv2.resize(thresh,(int(thresh.shape[1]/1),int(thresh.shape[0]/1))))
            thresh=thresh.reshape(thresh.shape[0],thresh.shape[1],1)
            #im=thresh[y1[i]:y2[i],x1[i]:x2[i],0]
            im,a,b,c=preprocessing(thresh[y1[i]:y2[i],x1[i]:x2[i],0],s_binary_thresh)[0:4]
            boundaries_test.append([a,b,c])
            lower_b.append(a)
            upper_b.append(b)
            original_l.append(c)
            cv2.imwrite('ocr_character/%d.jpg'%i,im)
            
        #image_from_array = Image.fromarray(im)
        #size_image = image_from_array.resize((30,30))
        size_image=transform.resize(im, (30, 30))
        data.append(np.array(size_image))
        #cv2.imwrite('ocr_training/images/%d.jpg'%number,im)
        cv2.rectangle(img_orig,(x1[i],y1[i]),(x2[i],y2[i]),(90,0,255),1)
        

    data=np.array(data)
    #data = data.astype('float32')/255
    if(method!='cnn'):
        data=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
        #if(method=='knn'):
            #data=nca.transform(data)

        
        
    #data=data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
    #img_orig=cv2.resize(img_orig,(int(img_orig.shape[1]/2),int(img_orig.shape[0]/2)))
    #img_orig=cv2.resize(img_orig,(int(img_orig.shape[1]/1),int(img_orig.shape[0]/1)))
    cv2.imshow('segmentation',cv2.resize(img_orig,(int(img_orig.shape[1]/2),int(img_orig.shape[0]/2))))
    cv2.imwrite('segmentation.jpg',cv2.resize(img_orig,(int(img_orig.shape[1]/1),int(img_orig.shape[0]/1))))
    img_orig=cv2.resize(img_orig,(int(img_orig.shape[1]/2),int(img_orig.shape[0]/2)))
    
    
    if(method=='cnn' or method=='ann'):
        #predictions=model.predict_classes(data)
        #probabilities=model.predict_proba(data)
        y_pred=model.predict_classes(data)
        y_prob=model.predict_proba(data)
        
        
    else:
        #predictions=model.predict(data)
        #probabilities=model.predict_proba(data)
        y_pred=model.predict(data)
        y_prob=model.predict_proba(data)
    text=''
    j=0
    
    extra_correction(predictions=y_pred,probabilities=y_prob,boundaries=boundaries_test,alphabet=alphabet)
    
    last_text=''

    diff=[]
    for i in range(1,len(x1)):
        if(x1[i]>x1[i-1]):
            diff.append(x1[i]-x2[i-1])
    diff=np.mean(diff)
    length=len(y_pred)
    for i in range(length):
        last_text+=alphabet[y_pred[i]]
        if(i!=(length-1)):
            space=round((x1[i+1]-x2[i])/(diff))
            if(space>1.95):
                last_text+=' '
            elif(x1[i+1]<x1[i]):
                last_text+='\n'
    print(last_text)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

        
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif prob < rdn < 2*prob:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output      
    
    