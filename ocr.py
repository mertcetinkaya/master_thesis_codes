import cv2
import random
import ocr_lib
random.seed(0)

#%%

#text='0123456789'
#alphabet=['0','1','2','3','4','5','6','7','8','9']
alphabet='AaBbCcÇçDdEeFfGgĞğHhIıİiJjKkLlMmNnOoÖöPpQqRrSsŞşTtUuÜüVvWwXxYyZz0123456789-.,;:?!`*()'
#text='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,;:?!`*()'
#alphabet=transform_text_to_alphabet(text)

tnr_0=cv2.imread('ocr_images/n_tnr_training_0.jpg')
tnr_1=cv2.imread('ocr_images/n_tnr_training_1.jpg')
tnr_2=cv2.imread('ocr_images/n_tnr_training_2.jpg')
tnr_2_blurred=cv2.blur(tnr_2,(3,3))
tnr_3=cv2.imread('ocr_images/n_tnr_training_3.jpg')
tnr_3_blurred=cv2.blur(tnr_3,(3,3))
courier_0=cv2.imread('ocr_images/n_courier_training_0.jpg')
courier_1=cv2.imread('ocr_images/n_courier_training_1.jpg')
courier_2=cv2.imread('ocr_images/n_courier_training_2.jpg')
courier_2_blurred=cv2.blur(courier_2,(3,3))
courier_3=cv2.imread('ocr_images/n_courier_training_3.jpg')
courier_3_blurred=cv2.blur(courier_3,(3,3))
bahnschrift_0=cv2.imread('ocr_images/n_bahnschrift_training_0.jpg')
bahnschrift_1=cv2.imread('ocr_images/n_bahnschrift_training_1.jpg')
bahnschrift_2=cv2.imread('ocr_images/n_bahnschrift_training_2.jpg')
bahnschrift_2_blurred=cv2.blur(bahnschrift_2,(3,3))
bahnschrift_3=cv2.imread('ocr_images/n_bahnschrift_training_3.jpg')
bahnschrift_3_blurred=cv2.blur(bahnschrift_3,(3,3))
arial_0=cv2.imread('ocr_images/n_arial_training_0.jpg')
arial_1=cv2.imread('ocr_images/n_arial_training_1.jpg')
arial_2=cv2.imread('ocr_images/n_arial_training_2.jpg')
arial_2_blurred=cv2.blur(arial_2,(3,3))
arial_3=cv2.imread('ocr_images/n_arial_training_3.jpg')
arial_3_blurred=cv2.blur(arial_3,(3,3))
cambria_0=cv2.imread('ocr_images/n_cambria_training_0.jpg')
cambria_1=cv2.imread('ocr_images/n_cambria_training_1.jpg')
cambria_2=cv2.imread('ocr_images/n_cambria_training_2.jpg')
cambria_2_blurred=cv2.blur(cambria_2,(3,3))
cambria_3=cv2.imread('ocr_images/n_cambria_training_3.jpg')
cambria_3_blurred=cv2.blur(cambria_3,(3,3))
verdana_0=cv2.imread('ocr_images/n_verdana_training_0.jpg')
verdana_1=cv2.imread('ocr_images/n_verdana_training_1.jpg')
verdana_2=cv2.imread('ocr_images/n_verdana_training_2.jpg')
verdana_2_blurred=cv2.blur(verdana_2,(3,3))
verdana_3=cv2.imread('ocr_images/n_verdana_training_3.jpg')
verdana_3_blurred=cv2.blur(verdana_3,(3,3))
#arial_0=cv2.imread('ocr_images/n_arial_training_0.jpg')
#arial_1=cv2.imread('ocr_images/n_arial_training_1.jpg')
#arial_2=cv2.imread('ocr_images/n_arial_training_2.jpg')
#arial_2_blurred=cv2.blur(arial_2,(3,3))
#arial_3=cv2.imread('ocr_images/n_arial_training_3.jpg')
#arial_3_blurred=cv2.blur(arial_3,(3,3))


#test_image=cv2.imread('consolas_test_2.jpg')
#test_image=cv2.imread('ocr_images/courier_test_1.jpg')
#test_image=cv2.imread('ocr_images/tarama.jpg')
#test_image=ocr_lib.sp_noise(test_image,0.01)
#test_image=cv2.blur(test_image,(4,4))
#cv2.imwrite('test_image.jpg',test_image)

training_image_list=[tnr_0,tnr_1,tnr_2,tnr_2_blurred,tnr_3,tnr_3_blurred,courier_0,courier_1,courier_2,courier_2_blurred,courier_3,courier_3_blurred,
                     bahnschrift_0,bahnschrift_1,bahnschrift_2,bahnschrift_2_blurred,bahnschrift_3,bahnschrift_3_blurred,
                     cambria_0,cambria_1,cambria_2,cambria_2_blurred,cambria_3,cambria_3_blurred,
                     verdana_0,verdana_1,verdana_2,verdana_2_blurred,verdana_3,verdana_3_blurred,
                     arial_0,arial_1,arial_2,arial_2_blurred,arial_3,arial_3_blurred]
random.seed(0)
tr_list=random.sample(training_image_list,int(len(training_image_list)*1))
ocr_lib.produce_training_data(training_image_list=tr_list,alphabet=alphabet,binary_thresh=200)


test_image=cv2.imread('ocr_images/consolas_test_1.jpg')
#test_image=cv2.imread('before_seg_7.jpg')


#test_image=cv2.blur(test_image,(5,5))
#test_image=cv2.imread('ocr_images/tarama.jpg')
#test_image=cv2.imread('consolas_test_1.jpg')
#test_image=ocr_lib.sp_noise(test_image,0.02)
cv2.imwrite('test_image.jpg',test_image)
ocr_lib.ocr(method='lr',alphabet=alphabet,test_image=test_image,f_binary_thresh=200,s_binary_thresh=200,
     training_image_list=tr_list,blurring=1)  

# ocr_lib.ocr(method='lr',alphabet=alphabet,test_image=test_image,f_binary_thresh=200,s_binary_thresh=200,
#     training_image_list=tr_list,blurring=1)  

#%%
image=cv2.imread('consolas_test_1.jpg')

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], -1, (255,255,255), 2)
    
    
# Remove vertical
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
# detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
# cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(image, [c], -1, (255,255,255), 2)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)[1]
thresh = cv2.medianBlur(thresh, 1)



cv2.imshow('thresh',thresh)
cv2.waitKey()
cv2.destroyAllWindows()



