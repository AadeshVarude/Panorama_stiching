import numpy as np
import cv2

def convol(img,kernel):
    # Convolution of image matrix with 3X3 kernel
    padx = 1
    pady = 1
    paddedimg = np.zeros((y + 2 * pady, x + 2 * padx))
    paddedimg[pady:-pady, padx:-padx] = img[:]
    opimg = np.zeros((y,x))
    for i in range(0, y):
        for j in range(0, x):
            opimg[i, j] = np.sum(kernel * paddedimg[i:i + 3, j:j + 3])

    return opimg

# Image input and preprocessing
img = cv2.imread('chessboard.png',0)
img = cv2.GaussianBlur(img,(5,5),0)
y, x = img.shape
img = cv2.normalize(img,img,alpha =  0,beta = 255, norm_type = cv2.NORM_MINMAX)


# Sobel partial differentiation operation
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

cv2.imwrite('sobelx.png',sobelopx)
cv2.imwrite('sobely.png',sobelopy)


# Calculating Structure Tensor
Ixx = sobelopx * sobelopx
Ixy = sobelopx * sobelopy
Iyx = sobelopy * sobelopx
Iyy = sobelopy * sobelopy

# Calculating windowed derivatives
rect_filter = 1/8*np.array([[1,1,1],[1,1,1],[1,1,1]])
wind_Ixx = convol(Ixx, rect_filter)
wind_Ixy = convol(Ixy, rect_filter)
wind_Iyx = convol(Iyx, rect_filter)
wind_Iyy = convol(Iyy, rect_filter)

# Harris response calculation
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


# Thresholding according to Harris response value
finalimg = np.zeros((x,y))
for i in range(0, x):
    for j in range(0, y):
        if R[i,j] > 0: # Corner
            finalimg[i,j] = 255
cv2.imwrite('final.png', finalimg)


