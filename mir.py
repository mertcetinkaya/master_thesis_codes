import mir_lib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,MaxPooling2D
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import load_model
import os


#%%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set(font_scale=1.1)
x=sns.heatmap(np.array([[34203,5545],[2260,13498]]),annot=True,fmt='g')
plt.savefig('hm_new.jpg')



#%%

#mir_lib.produce_data([4,8,12],32,'malaria',2,'training')
#mir_lib.produce_data([4,8,12],32,'malaria',2,'test')

mir_lib.function('nb',[4,8,12],2,1,'malaria')


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

edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
members=[]
directory='malaria'
data=mir_lib.cnn_produce_data(directory,edge_length)
#%%
model_count=2
epochs=25

for i in range(1,model_count+1): 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
    #model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer='he_uniform',activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3),kernel_initializer='he_uniform', activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    #model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    #model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
        
    
    model.add(Flatten())
    model.add(Dense(128,kernel_initializer='he_uniform', activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    #learning_rate=0.001
    #decay_rate = learning_rate / epochs
    #opt = Adam(lr=0.001, decay=learning_rate/epochs)
    #opt = Adam(lr=0.001)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model = load_model('best_model_'+str(i)+'.h5')
    #os.remove('best_model_'+str(i)+'.h5')
    a,b,c,d=mir_lib.cnn(model,edge_length,epochs,directory,i,{0:0.5,1:0.5},0.5,training_val_test_data=data,checkpoint=True)
    training_acc.append(a)
    val_acc.append(b)
    test_acc.append(c)
    members.append(d)
#%%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=False)
    
#%%
model_number=1
filepath="best_model_weights_"+str(model_number)+".hdf5"
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(edge_length,edge_length,3)))
#model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# y_training_predicted=np.where(model.predict_proba(data[0])[:,0] > 0.224,0,1)
# y_val_predicted=np.where(model.predict_proba(data[1])[:,0] > 0.224,0,1)
# y_test_predicted=np.where(model.predict_proba(data[2])[:,0] > 0.224,0,1)


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

#%%

model_count=5

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
mir_lib.cnn_probability_moving(model,data,'breast_cancer')


#%%
#transfer learning

edge_length=32
training_acc=[]
val_acc=[]
test_acc=[]
directory='malaria'
data=mir_lib.cnn_produce_data(directory,edge_length)
model_count=2
epochs=5

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.applications import MobileNet
for i in range(1,model_count+1): 
    base_model=MobileNet(weights='imagenet',include_top=False)
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(64,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    #for layer in base_model.layers:
    #	layer.trainable = False
    
    #for layer in model.layers[:20]:
    #    layer.trainable=False
    #for layer in model.layers[20:]:
    #    layer.trainable=True
    #opt = Adam(lr=0.005)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    a,b,c=mir_lib.cnn(model,edge_length,epochs,directory,i,{0:0.5,1:0.5},0.5,training_val_test_data=data,transfer_learning=True)
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
plt.savefig(directory+'/acc_by_model.jpg')
plt.show()

#%%
import cv2
#%%
image=cv2.imread(r"C:\Users\mertc\Desktop\boun\tez hk\data\malaria\binary_images\training_all\1\C33P1thinF_IMG_20150619_114756a_cell_179.png")
cv2.imwrite('orig_m.jpg',image)
cv2.imwrite('blur_m.jpg',cv2.blur(image,(5,5)))
cv2.imwrite('rotate180_m.jpg',cv2.rotate(image,cv2.ROTATE_180))
cv2.imwrite('rotate90cw_m.jpg',cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
cv2.imwrite('rotate90ccw_m.jpg',cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE))
#%%

image=data[0][0]
#image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
#image=image.reshape(image.shape[0],image.shape[1],1)
#image=tsr_lib.preprocess_img(image,32)
cv2.imshow('xx',cv2.resize(image,(200,200)))
cv2.waitKey()
cv2.destroyAllWindows()

#%%

data_train_gray=[]
data_val_gray=[]
data_test_gray=[]

for i in range(len(data[0])):
    data_train_gray.append(cv2.cvtColor(np.float32(data[0][i]),cv2.COLOR_BGR2GRAY).flatten())
for i in range(len(data[1])):
    data_val_gray.append(cv2.cvtColor(np.float32(data[1][i]),cv2.COLOR_BGR2GRAY).flatten())
for i in range(len(data[2])):
    data_test_gray.append(cv2.cvtColor(np.float32(data[2][i]),cv2.COLOR_BGR2GRAY).flatten())
    
#%%
import cv2
directory='malaria'
edge_length=8
data=mir_lib.cnn_produce_data(directory,edge_length)

data_train_gray=[]
data_val_gray=[]
data_test_gray=[]

for i in range(len(data[0])):
    data_train_gray.append(cv2.cvtColor(np.float32(data[0][i]),cv2.COLOR_BGR2GRAY).flatten())
for i in range(len(data[1])):
    data_val_gray.append(cv2.cvtColor(np.float32(data[1][i]),cv2.COLOR_BGR2GRAY).flatten())
for i in range(len(data[2])):
    data_test_gray.append(cv2.cvtColor(np.float32(data[2][i]),cv2.COLOR_BGR2GRAY).flatten())


from sklearn.neighbors import NeighborhoodComponentsAnalysis
import seaborn as sns
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(data_train_gray, np.argmax(data[3],axis=1))
X_reduced_nca = nca.transform(data_train_gray)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = np.argmax(data[3],axis=1)
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import  GridSearchCV
def KNN_Best_Params(x_train, x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = "accuracy")
    grid.fit(x_train, y_train)
    
    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
    print()
    print("CM Test: ",cm_test)
    print("CM Train: ",cm_train)
    
    return grid

X_train_nca = X_reduced_nca
X_test_nca = nca.transform(data_val_gray)
Y_train_nca = np.argmax(data[3],axis=1)
Y_test_nca = np.argmax(data[4],axis=1)

grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("%i-Class classification (k = %i, weights = '%s')"
#           % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

#%%

edge_length=32
directory='breast_cancer'
data=mir_lib.cnn_produce_data(directory,edge_length)

#%%

data_train_gray=[]
data_val_gray=[]
data_test_gray=[]

for i in range(len(data[0])):
    data_train_gray.append(np.float32(data[0][i,:,:,0]).flatten())
for i in range(len(data[1])):
    data_val_gray.append(np.float32(data[1][i,:,:,0]).flatten())
for i in range(len(data[2])):
    data_test_gray.append(np.float32(data[2][i,:,:,0]).flatten())


from sklearn.neighbors import NeighborhoodComponentsAnalysis
import seaborn as sns
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(data_train_gray, np.argmax(data[3],axis=1))
X_reduced_nca = nca.transform(data_train_gray)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = np.argmax(data[3],axis=1)
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")



#%%
import cv2
edge_length=32
directory='malaria'
data=mir_lib.read_data(directory,edge_length)
#gercek veya gri okumak guzel ayirt edilmeli. nca gri ile. hsv ve canny gercek ile
#%%
img=data[0][2].copy()
cv2.imshow('orig',cv2.resize(img,(200,200)))
canny=cv2.Canny(img,100,100)
cv2.imshow('canny1',cv2.resize(canny,(200,200)))
canny = cv2.GaussianBlur(canny, (3,3),0) 
#canny=cv2.medianBlur(canny, 5)
#canny=cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY)[1]
#canny = cv2.erode(canny, None, iterations = 1)
#canny = cv2.dilate(canny, None, iterations = 1)
cv2.imshow('canny2',cv2.resize(canny,(200,200)))
(contours,_) = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) > 0:
    
    # en buyuk konturu al
    #c = min(contours, key = cv2.contourArea)
    for i in contours:
        # dikdörtgene çevir 
        rect = cv2.minAreaRect(i)
        
        ((x,y), (width,height), rotation) = rect
        
        s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
        print(s)
        
        # kutucuk
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        
        # moment
        #M = cv2.moments(c)
        #center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        
        # konturu çizdir: sarı
        if(20<width<25 and 20<height<25):
            cv2.drawContours(img, [box], 0, (0,255,255),2)
        
        # merkere bir tane nokta çizelim: pembe
        #cv2.circle(img, center, 5, (255,0,255),-1)




cv2.imshow('x',cv2.resize(img,(200,200)))

cv2.waitKey()
cv2.destroyAllWindows()

#%%
#HSV
img=data[0][2].copy()
cv2.imshow('orig',cv2.resize(img,(200,200)))
#img = cv2.imread("malaria/binary_images/training_all/1/C33P1thinF_IMG_20150619_114756a_cell_179.png")
# Convert BGR to HSV
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
#lower_blue = np.array([50,50,150])
#upper_blue = np.array([255,255,200])
## Threshold the HSV image to get only blue colors
#mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
#res = cv2.bitwise_and(img,img, mask= mask)
#cv2.imshow('x',img)
#cv2.imshow('xx',res)
#cv2.waitKey()
#cv2.destroyAllWindows()

blueLower = (84,  98,  0)
blueUpper = (179, 255, 255)

blueLower = (153,  91,  100)
blueUpper = (255, 255, 255)

# blur
#blurred = cv2.GaussianBlur(img, (1,1), 0) 

# hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image",cv2.resize(hsv,(200,200)))

# mavi için maske oluştur
mask = cv2.inRange(hsv, blueLower, blueUpper)
cv2.imshow("mask Image",cv2.resize(mask,(200,200)))
# maskenin etrafında kalan gürültüleri sil
#mask = cv2.erode(mask, None, iterations = 2)
#mask = cv2.dilate(mask, None, iterations = 2)
#cv2.imshow("Mask + erozyon ve genisleme",cv2.resize(mask,(200,200)))

# farklı sürüm için
# (_, contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# kontur
(contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center = None

if len(contours) > 0:
    
    # en buyuk konturu al
    c = max(contours, key = cv2.contourArea)
    
    # dikdörtgene çevir 
    rect = cv2.minAreaRect(c)
    
    ((x,y), (width,height), rotation) = rect
    
    s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
    print(s)
    
    # kutucuk
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    # moment
    M = cv2.moments(c)
    center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    
    # konturu çizdir: sarı
    cv2.drawContours(img, [box], 0, (0,255,255),2)
    
    # merkere bir tane nokta çizelim: pembe
    #cv2.circle(img, center, 5, (255,0,255),-1)
    
    
    
cv2.imshow('x',cv2.resize(img,(200,200)))
cv2.waitKey()
cv2.destroyAllWindows()




#%%
from sklearn.model_selection import train_test_split

x=np.concatenate([data[0].reshape(data[0].shape[0],-1),data[1].reshape(data[1].shape[0],-1)])
#x=x/255
y=np.concatenate([np.argmax(data[3],axis=1),np.argmax(data[4],axis=1)])

#idx = np.random.choice(np.arange(len(x)), 10000, replace=False)
#x = x[idx]
#y = y[idx]

#x, _, y, _= train_test_split(x,y,stratify=y,train_size=25000,random_state=42)

#%%
#from sklearn.manifold import TSNE

# tsne = TSNE(random_state=0,)
# tsne_results = tsne.fit_transform(x)

# tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])

# plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=y)
# plt.show()


from sklearn.neighbors import NeighborhoodComponentsAnalysis
import seaborn as sns
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x, y)
X_reduced_nca = nca.transform(x)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA for Breast Cancer Data")


#%%


import seaborn as sns

x=np.concatenate([data[0].reshape(data[0].shape[0],-1),data[1].reshape(data[1].shape[0],-1)])
x=np.mean(x,axis=1)
#x=x/255
y=np.concatenate([np.argmax(data[3],axis=1),np.argmax(data[4],axis=1)])
#%%

#x=x.flatten()
#y=np.repeat(y,1024)

#idx=np.where(x!=0)
#x=x[idx]
#y=y[idx]


mydata=list(zip(x,y))

#%%
sns.set_theme()
sns.displot(pd.DataFrame(mydata,columns=(['x','y'])), x=x,hue=y, stat="density",kde=True)
plt.title('Density Plot and Histogram for Malaria Data')
plt.xlabel('Average Gray Pixel Value')
plt.ylabel('Density')