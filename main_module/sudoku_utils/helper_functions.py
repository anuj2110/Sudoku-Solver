from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image,debug = False):
    '''
    Function to find the sudoku puzzle in the image.
    image: a numpy array (image/video frame in which the puzzle is)
    debug: parameter to see the intermediate steps (default:False)
    '''
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3)

    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Puzzle thresh",thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea, reverse=True)

    puzzleCnt = None
    
    for c in cnts:
        peri = cv2.arcLength(c,True) # calculate the perimeter of the counters
        approx = cv2.approxPolyDP(c,0.02*peri,True) # We check if the given countour represents a square or not

        # if our approximated contour has four points, then we can
		# assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    if puzzleCnt is None:
        raise Exception(("Could not find sudoku puzzle outline. Try debugging your thresholding and contour steps."))

    if debug:
		# draw the contour of the puzzle on the image and then display
		# it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
    
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    
    return (puzzle, warped)


def extract_digit(cell, debug=False):
    '''
    Fucntion to extract the digits from the sudoku image
    cell: numpy array (cell of the sudoku puzzle)
    debug: parameter to see the intermediate steps (default:False)
    '''

    thresh = cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh = clear_border(thresh)


    if debug:
        cv2.imshow("Cell Thresh",thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) ==0:
        return None
    
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h,w) = thresh.shape

    precentFilled = cv2.countNonZero(mask)/float(w*h)

    if precentFilled<0.03:
        return None
    
    digit = cv2.bitwise_and(thresh,thresh,mask=mask)

    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    
    return digit

