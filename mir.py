import mir_lib
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout,MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.optimizers import Adam
import cv2
#%%
#hog feature data production, training and testing
mir_lib.produce_data([5,10,15],100,'malaria',2,'training')
mir_lib.produce_data([5,10,15],100,'malaria',2,'test')
mir_lib.function('nb',[5,10,15],2,1,'malaria')


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
#malaria cnn data production
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
directory='malaria'
data=mir_lib.cnn_produce_data(directory,edge_length)
#%%
#cnn training for malaria recognition
model_count=2
epochs=5

for i in range(1,model_count+1): 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    
    model.add(Flatten())
    model.add(Dense(128,kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    a,b,c,d=mir_lib.cnn(model,edge_length,epochs,directory,i,{0:0.5,1:0.5},0.5,training_val_test_data=data,checkpoint=False)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    members.append(d)
#%%
#model plotting
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=False)

#%%
#result visualization
model_count=2

fr=show_statistics([training_acc,val_acc,test_acc],['training','val','test'])


plt.plot(range(1,len(training_acc)+1),training_acc,'k.-')
plt.plot(range(1,len(training_acc)+1),val_acc,'b.-')
plt.xticks(np.arange(1,model_count+1,1))
plt.legend(labels=['training_acc','val_acc'])
plt.ylabel('accuracy')
plt.xlabel('model')
plt.title('accuracy by model')
plt.savefig(directory+'/acc_by_model.jpg')
plt.show()

#%%
#probability threshold moving
mir_lib.cnn_probability_moving(model,data,'malaria')

#%% 
#printing results in a detailed way without probability threshold moving
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

y_training_predicted=np.argmax(model.predict(data[0]),axis=1)
y_val_predicted=np.argmax(model.predict(data[1]),axis=1)
y_test_predicted = np.argmax(model.predict(data[2]),axis=1)

print('accuracy')
print(accuracy_score(y_training_predicted,np.argmax(data[3],axis=1)))
print(accuracy_score(y_val_predicted,np.argmax(data[4],axis=1)))
print(accuracy_score(y_test_predicted,data[5]))
print('precision')
print(precision_score(y_training_predicted,np.argmax(data[3],axis=1)))
print(precision_score(y_val_predicted,np.argmax(data[4],axis=1)))
print(precision_score(y_test_predicted,data[5]))
print('recall')
print(recall_score(y_training_predicted,np.argmax(data[3],axis=1)))
print(recall_score(y_val_predicted,np.argmax(data[4],axis=1)))
print(recall_score(y_test_predicted,data[5]))
print('f1 measure')
print(f1_score(y_training_predicted,np.argmax(data[3],axis=1)))
print(f1_score(y_val_predicted,np.argmax(data[4],axis=1)))
print(f1_score(y_test_predicted,data[5]))
    
#%%
#data augmentation example
image=cv2.imread(r"malaria\binary_images\training_all\1\C33P1thinF_IMG_20150619_114756a_cell_179.png")
cv2.imwrite('orig_m.jpg',image)
cv2.imwrite('blur_m.jpg',cv2.blur(image,(5,5)))
cv2.imwrite('rotate180_m.jpg',cv2.rotate(image,cv2.ROTATE_180))
cv2.imwrite('rotate90cw_m.jpg',cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
cv2.imwrite('rotate90ccw_m.jpg',cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE))


#%%
#reading malaria data
edge_length=32
directory='malaria'
data=mir_lib.read_data(directory,edge_length)
#%%
#malaria smear detection with canny edge detection
img=data[0][2].copy()
cv2.imshow('orig',cv2.resize(img,(200,200)))
canny=cv2.Canny(img,100,100)
cv2.imshow('canny1',cv2.resize(canny,(200,200)))
canny = cv2.GaussianBlur(canny, (3,3),0) 

cv2.imshow('canny2',cv2.resize(canny,(200,200)))
(contours,_) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    
    
    for i in contours:
        
        rect = cv2.minAreaRect(i)
        
        ((x,y), (width,height), rotation) = rect
        
        s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
        print(s)
        
        
        box = cv2.boxPoints(rect)
        box = np.int64(box)
                
        
        if(20<width<25 and 20<height<25):
            cv2.drawContours(img, [box], 0, (0,255,255),2)


cv2.imshow('x',cv2.resize(img,(200,200)))

cv2.waitKey()
cv2.destroyAllWindows()

#%%
#malaria smear detection with hsv masking
#HSV
img=data[0][2].copy()
cv2.imshow('orig',cv2.resize(img,(200,200)))

blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

blueLower = (153,  91,  100)
blueUpper = (255, 255, 255)

# hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image",cv2.resize(hsv,(200,200)))

# mavi için maske oluştur
mask = cv2.inRange(hsv, blueLower, blueUpper)
cv2.imshow("mask Image",cv2.resize(mask,(200,200)))

(contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center = None

if len(contours) > 0:
    
    c = max(contours, key = cv2.contourArea)
    
    rect = cv2.minAreaRect(c)
    
    ((x,y), (width,height), rotation) = rect
    
    s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
    print(s)
    
    
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    cv2.drawContours(img, [box], 0, (0,255,255),2)
    
   
cv2.imshow('x',cv2.resize(img,(200,200)))
cv2.waitKey()
cv2.destroyAllWindows()

#%%
#breast cancer cnn data production
edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
directory='breast_cancer'
data=mir_lib.cnn_produce_data(directory,edge_length)
#%%
#breast cancer training with checkpoint
model_count=1
epochs=5

for i in range(1,model_count+1): 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    
    model.add(Flatten())
    model.add(Dense(128,kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    a,b,c,d=mir_lib.cnn(model,edge_length,epochs,directory,i,{0:0.5,1:0.5},0.5,training_val_test_data=data,checkpoint=True)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    members.append(d)


#%%
#loading the best model
model_number=1
filepath="best_model_weights_"+str(model_number)+".hdf5"
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
#model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
#model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
    

model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
#learning_rate=0.001
#decay_rate = learning_rate / epochs
#opt = Adam(lr=0.001, decay=learning_rate/epochs)
#opt = Adam(lr=0.001)
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
#heatmap saving
import seaborn as sns
from sklearn.metrics import confusion_matrix
medical_image_directory='breast_cancer'
total_class_number=2
fig = plt.figure(figsize=(10, 8))
#fig.add_subplot(121)
sns.heatmap(confusion_matrix(y_pred=y_training_predicted,y_true=np.argmax(data[3],axis=1)), annot=True,fmt='g')
plt.title('training confusion matrix')
plt.ylabel('actual')
plt.xlabel('predicted')
plt.savefig(medical_image_directory+'/tr_conf'+str(model_number)+'.jpg')
plt.show()
fig = plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_pred=y_val_predicted,y_true=np.argmax(data[4],axis=1),labels=range(total_class_number)),  annot=True,fmt='g',annot_kws={"size":8})
plt.title('val confusion matrix')
plt.ylabel('actual')
plt.xlabel('predicted')
plt.savefig(medical_image_directory+'/val_conf'+str(model_number)+'.jpg')
plt.show()
fig = plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_pred=y_test_predicted,y_true=data[5]), annot=True,fmt='g')
plt.title('test confusion matrix')
plt.ylabel('actual')
plt.xlabel('predicted')
plt.savefig(medical_image_directory+'/test_conf'+str(model_number)+'.jpg')
plt.show()