# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 21:12:10 2021

@author: fernando pazos

This script trains a regressor algorithm to determine the position of the eyes of a person.
It uses as database the pictures of the BioIDFace database, available in
http://www.bioid.com/support/downloads/software/bioid-face-database.html
The dataset consists of 1521 gray level images with a resolution of 384×286 pixel. 
Each one shows the frontal view of a face of one out of 23 different test persons. For 
comparison reasons the set also contains manually set eye postions. The images are labeled 
“BioID_xxxx.pgm” where the characters xxxx are replaced by the index of the current image 
(with leading zeros). Similar to this, the files “BioID_xxxx.eye” contain the eye positions 
for the corresponding images.
The images are stored in single files using the portable gray map (pgm) data format. A pgm file 
contains 15 bytes of data header followed by the image data. These bytes are
"P5": Indicates that the data is in binary form
"384": widht of the pictures in pixels
"286": height of the picture in pixels
"255": maximum value of the allowable grey level
384*286 bytes with the gray level of each pixel. The data is stored line per line from top to 
bottom using one byte per pixel.

The files with the eye position are text files with two lines.
the first line is "#LX	LY	RX	RY"
the second line are the coordinates such as "223 103	154	102"

Please, change the pathes at the lines 54 and 63 if the files are in a folder different than
the one specified here.
"""

import numpy as np

#function that returns a string with the integer passed as argument with four characters, with as many leading '0' as necessary
def nome(i):
    nz=3
    s=i
    for j in range(3):
        s//=10
        if s!=0:
            nz-=1
    sr='0'*nz+str(i)
    return sr

#number of pictures in the data file
npictures=1521

# reading the data base to put the pixels of the pictures in a matrix denoted as Pic
nbytes=384*286+15     #number of bytes of every file: 15 bytes of header + 384*286 pixels
Pic=np.zeros((npictures,nbytes)).astype("uint8") 
for i in range(npictures):
 #change the path if the files are in a different folder   
 Pic[i]=np.fromfile('C:/Users/fernando/BioIDFace/BioID-FaceDatabase-V1.2/BioID_'+nome(i)+'.pgm', dtype=np.uint8, count=-1, sep='')

#deleting the header of each file
Pic=Pic[:,15:]

#reading the data base with the positions of the eyes to put the coordinates in a matrix denoted as Pos
Pos=np.zeros((npictures,4)).astype("uint16")
for i in range(npictures):
    #change the path if the files are in a different folder
    f=open('C:/Users/fernando/BioIDFace/BioID-FaceDatabase-V1.2/BioID_'+nome(i)+'.eye')
    f.readline()            #read the first line with #LX LY RX RY
    s=f.readline()          #read the second line with the four coordinates
    k=''
    #num=[0]*4
    j=0
    for l in range(len(s)):
        if s[l]!='\t' and s[l]!='\n':
            k+=s[l]
        else:
            Pos[i,j]=int(k)
            k=''
            j+=1
            
#setting the training matrices
X_train=Pic
Y_train=Pos         

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor

# Fit estimators
ESTIMATORS = {
    "Extra Tree": ExtraTreesRegressor(n_estimators=80, max_features=500, random_state=0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Support Vector Machine": MultiOutputRegressor(SVR()),
    "K Nearest Neighbors": KNeighborsRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge CV": RidgeCV(),
    "PLS Regression": PLSRegression(n_components=8),
    "Multi-layer Perceptron": MLPRegressor(random_state=1, max_iter=300),}


reg=ESTIMATORS[regressor.value].fit(X_train,Y_train)

print("Regresor used: ",regressor.value)

#choosing randomly only one picture to test
print("Picture from the data base randomly chosen to test")
numpic=int(np.random.uniform(0,npictures))
print("Image tested:",numpic)
X_test=Pic[numpic,:].reshape(1, -1)
Y_test=Pos[numpic,:].reshape(1, -1)

#testing the picture chosen
Y_predict=reg.predict(X_test)

print("Result of the prediction", Y_predict)
print("Coordinates values",Y_test)
print("Error:",np.linalg.norm(Y_predict-Y_test))
