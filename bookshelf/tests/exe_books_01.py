import cv2 as cv
from bookshelf.function.display import display
from bookshelf.function.preprocessing import preprocessing
from bookshelf.function.contours import contours
from bookshelf.function.watershed import watershed
from bookshelf.function.img_write import img_write


'''
function_list = [
    'display', 
    'preprocessing', 
    'cv.threshold',
]
'''


# load image
path = 'C:/Object-detection/EyeCandy/img/bookshelf_04.jpg'
img_color = cv.imread(path)
display('img_color', img_color)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
display('img_gray', img_gray)

# deal with no-background colors
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
display('sep_threshold', thresh)
img_color = preprocessing(img_color, 3, 3)
img_write('1_thresh', img_color)

# deal with similar-background colors
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)
display('sep_threshold', thresh)
img_color = preprocessing(img_color)
img_write('2_thresh', img_color)

# do contour
path = 'C:/Object-detection/EyeCandy/img/img_book_only.png'
img_book_only = cv.imread(path)
img_color = contours(img_book_only, img_color)
img_write('3_contour', img_color)

# do watershed
img_watershed = watershed(img_color)
display('watershed img', img_watershed)
img_write('watershed', img_watershed)

cv.destroyAllWindows()