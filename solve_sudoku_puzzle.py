from main_module.sudoku_utils.helper_functions import extract_digit
from main_module.sudoku_utils.helper_functions import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()

ap.add_argument('-m','--model',required = True,help="Path to the model file")
ap.add_argument('-i','--image',required=True,help='Path to the image file')
ap.add_argument('-d','--debug',help='Parameter to see the intermediate stages',default=-1,type=int)

args = vars(ap.parse_args())

print("LOADING MODEL ...")
model = load_model(args['model'])

print("LOADING IMAGE ...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

(puzzleImage,warped) = find_puzzle(image,debug=args['debug']>0)
board = np.zeros((9,9),dtype="int")

stepX = warped.shape[1]//9
stepY = warped.shape[0]//9

cellLocs = []

for y in range(0,9):

    row =[]

    for x in range(0,9):

        startX = x*stepX
        startY = y*stepY

        endX = (x+1)*stepX
        endY = (y+1)*stepY

        row.append((startX,startY,endX,endY))

        cell = warped[startY:endY,startX:endX]
        digit = extract_digit(cell,debug=args['debug']>0)

        if digit is not None:
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            pred = model.predict(roi)
            board[y,x] = np.argmax(pred)
    
    cellLocs.append(row)


print("SUDOKU OCR'd Sudoku Board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

print("SOLVING SUDOKU ...")
solution = puzzle.solve()
solution.show()

for (cellRow,boardRow) in zip(cellLocs,solution.board):
    for (box,digit) in zip(cellRow,boardRow):
        startX,startY,endX,endY = box

        textX = int((endX-startX)*0.33)
        textY = int((endY-startY)*0.66)
        textX+=startX
        textY+=startY
        cv2.putText(puzzleImage, str(digit), (textX, textY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)    

cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)

