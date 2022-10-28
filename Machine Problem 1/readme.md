This folder contains 4 image samples as input to the program, 1 .ipynb file, and 1 .py file. Both .ipynb and .py files contain the functions and codes to remove the projective distortion of the image.

The Jupyter notebook prints the source points, destination points, homography matrix, and other small details details that are used to run the program. The python file is a copy of the Jupyter notebook, except that it only reads the image, lets user select source points, and return the projected image. User can also select other image files in the Python file by typing the file name/path in the input.

Libraries used: Numpy, OpenCV, PIL
