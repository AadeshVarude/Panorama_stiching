import numpy as np
import cv2

def convol(img,kernel):
    padx = 1
    pady = 1
    paddedimg = np.zeros((y + 2 * pady, x + 2 * padx))
    paddedimg[pady:-pady, padx:-padx] = img[:]
    opimg = np.zeros((y,x))
    for i in range(0, y):
        for j in range(0, x):
            opimg[i, j] = np.sum(kernel * paddedimg[i:i + 3, j:j + 3])

    return opimg

img = cv2.imread('lenna.png',0)
img = cv2.GaussianBlur(img,(5,5),0)
y, x = img.shape
img = cv2.normalize(img,img,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
# print(img)
# Sobel operation
padx = 1
pady = 1
paddedimg = np.zeros((y + 2 * pady, x + 2 * padx))
paddedimg[pady:-pady,padx:-padx] = img[:]

sobelx = np.array([[1,0,-1],
                   [2,0,-2],
                   [1,0,-1]])

sobely = np.array([[1,2,1],
                   [0,0,0],
                   [-1,-2,-1]])

sobelopx = convol(img,sobelx)
sobelopy = convol(img,sobely)

# outputx = int(x - 2 + 2 * padx)
# outputy = int(y - 2 + 2 * pady)
# sobelopx = np.zeros((outputy, outputx))
# sobelopy = np.zeros((outputy, outputx))
# for i in range(0, sobelopx.shape[0]):
#     for j in range(0, sobelopx.shape[1]):
#         sobelopx[i, j] = np.sum(sobelx * paddedimg[i:i + 3, j:j + 3])
#         sobelopy[i, j] = np.sum(sobely * paddedimg[i:i + 3, j:j + 3])

# sobelopx = cv2.normalize(sobelopx,sobelopx,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
# sobelopy = cv2.normalize(sobelopy,sobelopy,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
cv2.imwrite('sobelx.png',sobelopx)
cv2.imwrite('sobely.png',sobelopy)
# print(sobelopx)

Ixx = sobelopx * sobelopx
Ixy = sobelopx * sobelopy
Iyx = sobelopy * sobelopx
Iyy = sobelopy * sobelopy

rect_filter = 1/8*np.array([[1,1,1],[1,1,1],[1,1,1]])

wind_Ixx = convol(Ixx, rect_filter)
wind_Ixy = convol(Ixy, rect_filter)
wind_Iyx = convol(Iyx, rect_filter)
wind_Iyy = convol(Iyy, rect_filter)


# Ixx = np.zeros(sobelopx.shape)
# Ixy = np.zeros(sobelopx.shape)
# Iyx = np.zeros(sobelopx.shape)
# Iyy = np.zeros(sobelopx.shape)
# Ixx = cv2.normalize(sobelopx * sobelopx,Ixx,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
# Ixy = cv2.normalize(sobelopx * sobelopy,Ixy,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
# Iyx = cv2.normalize(sobelopy * sobelopx,Iyx,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
# Iyy = cv2.normalize(sobelopy * sobelopy,Iyy,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)
cv2.imwrite('xx.png',Ixx)
cv2.imwrite('xy.png',Ixy)
cv2.imwrite('yx.png',Iyx)
cv2.imwrite('yy.png',Iyy)
# print(Ixx)
M = np.zeros((2,2))
k = 0.04
R = np.zeros(img.shape)
for i in range(0, x):
    for j in range(0, y):
        M[0, 0] = wind_Ixx[i,j]
        M[0, 1] = wind_Ixy[i, j]
        M[1, 0] = wind_Iyx[i, j]
        M[1, 1] = wind_Iyy[i, j]

        determinant = np.linalg.det(M)
        trace = np.trace(M)
        R[i, j] = determinant - (k*(trace**2))

# R = cv2.normalize(R,R,alpha =  0,beta = 127, norm_type = cv2.NORM_MINMAX)
finalimg = np.zeros((x,y))
for i in range(0, x):
    for j in range(0, y):
        if R[i,j] > 0:
            finalimg[i,j] = 255
cv2.imwrite('final.png', finalimg)
print(R)


