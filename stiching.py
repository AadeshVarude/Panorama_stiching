import cv2
import numpy as np
import os
import math

path = os.getcwd()

image_path = path + '/images/set1' 
os.chdir(image_path)

def required_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    image,contour,heic = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contour[0]
    cnt = cnt.reshape(-1,2)
    x_min = np.min(cnt[:,0],axis = 0) 
    y_min = np.min(cnt[:,1],axis = 0)
    x = np.max(cnt[:,0],axis = 0)
    y = np.max(cnt[:,1],axis = 0)
    img2 = np.zeros((y,x,3),np.uint8)
    img2 = img[y_min:y,x_min:x]
    # cv2.waitKey(0)
    return img2

def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img


listdir = os.listdir()
listdir = sorted(listdir)

img1 = cv2.imread(listdir[0])
#img1 = cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

canvas = np.zeros((img1.shape[0]*4,img1.shape[1]*5,img1.shape[2]),np.uint8)

canvas[150:150+img1.shape[0],100:100+ img1.shape[1]] = img1
img3 = canvas.copy()
i = 1

exten_list = ['.jpg','jpeg','.bmp','.png']
features_extractor = "SIFT"
print(listdir)
# read only when extension is of type ['.jpg','jpeg',".bmp",'.png']
while i < len(listdir):
        img1 = img3
        if  listdir[i][-4:] not in exten_list:
            continue
        img2 = cv2.imread(listdir[i])
     #  img2 = cv2.resize(img2,None, fx=0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC) 
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
    
        if features_extractor.upper() == "SIFT":

            kp1,desc1 = sift.detectAndCompute(gray1,None)
            kp2,desc2 = sift.detectAndCompute(gray2,None)
        else:
            pass

        bf = cv2.BFMatcher(crossCheck=False)
        matches = bf.knnMatch(desc1,desc2,k=2)
        good = []
        pt1 = []
        pt2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        good = sorted(good,key = lambda x:x.distance)
        good = good
        pt1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pt2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        print(listdir[i],pt1.shape,pt2.shape)
        if pt1.shape[0]  <= 20 or pt2.shape[0] <= 20:
            i += 1
            continue
        else:
             pass 
        
        H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,ransacReprojThreshold=4.0)
        if H is not None:
            H = np.linalg.inv(H)
            img3 = cv2.warpPerspective(img2,H,(canvas.shape[1],canvas.shape[0]))
            canvas = image_stiching(canvas,img3)
        i += 1
        if cv2.waitKey(2) == 2:
            break

canvas = required_img(canvas)
cv2.imshow('panorama',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
