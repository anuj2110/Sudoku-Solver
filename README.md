# Sudoku Puzzle Solver with Computer Vision and Deep Learning
---

In this project we try to build an end to end system which can solve the sudoku puzzle given the image of the puzzle by leveraging Computer Vision and Deep Learning

## Libraries used:
1. Tensorflow Keras
2. OpenCV
3. imutils
4. NumPy
5. py_sudoku

## Steps Involved are:

1. Make a model for digit recognition with tensorflow, keras and MNIST dataset.

2. Extract the board from the image. From the extracted board, extract each cell from the image and check if it has a digit or not. If yes, then recognize it with model built in 1 else continue.

3. Repeat step 2 for all the cells and save the results in an matrix. 

4. After this using the py_sudoku solver solve the sudoku puzzle and display the results on the board.
