# Eye-detection-using-ML
Machine learning app to detect eye position

This a Delphi coded application which uses Python4Delphi components (see https://github.com/pyscripter/python4delphi by pyscripter)
At the left part of the screen a top memo allows to write a python script, which can be executed using the botton memo as output console.
The python script can be loaded from a file, saved, and the memo can be cleared also.
This application requires to run the script BioIdFace_train.py. This program trains a Machine Learning algorithm to detect the position 
of the eyes of the people portrayed in the images in the database files used. These images can be downloaded from the BioId database
(https://www.bioid.com/facedb/). The dataset consists of 1521 gray level images with a resolution of 384×286 pixel. Each one shows the frontal 
view of a face of one out of 23 different test persons. For comparison reasons the set also contains manually set eye postions.
The algorithm used can be selected in the botton combobox (Extra Tree, Decision Tree, Support Vector Machine, K Nearest Neighbors, Linear Regression
Ridge CV, PLS Regression and Multi-layer Perceptron). Hiper-parameters must be defined directly in the script. This script also requires to modify 
the lines 54 and 63 substituting the path there written by the one where the database files are stored in your computer.

Once the algorithm is trained, a PGM file with 384 columns, 286 rows and 256 gray levels can be opened and loaded at the right part of the screen.
This image is the test set of the algorithm. The script to test can be executed by clicking in the button "Test Eye Position". The coordinates 
of the eyes calculated by the algorithm will be printed in the output console and light green crosses will be printed on the images in the positions 
calculated.

In the FMX version of the application, the web cam can be turned on, the image taken by it appears in a little picture at the top of the screen,
and this image can be photographed to put it in the boton picture of the screen in a suitable format.

Of course, the algorithm can be changed. In this case it is not necessary to execute all the python script again, only those sentences where the 
algorithm selected is trained (you can delete the first sentences of the script where the files of the database are loaded).


![BioIdFaceApp](https://user-images.githubusercontent.com/37451727/144942643-78b1cb0c-dac1-4c20-932f-a5a88da14962.png)
