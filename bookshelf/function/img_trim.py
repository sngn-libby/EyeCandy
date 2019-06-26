# Reference Code :  https://bit.ly/2x9YTVB

import cv2 as cv


def img_trim (self, img_color, x, y, w, h):

    self.x = x; self.y = y; # 자르고 싶은 지점의 x좌표와 y좌표 지정
    self.w = w; self.h = h; # x로부터 width, y로부터 height를 지정
    img_trim = img_color[y:y+h, x:x+w] # trim한 결과를 img_trim에 담는다
    cv.imwrite('org_trim.jpg',img_trim) # org_trim.jpg 라는 이름으로 저장

    return img_trim # 필요에 따라 결과물을 리턴

# org_image = cv.imread('test.jpg') #test.jpg 라는 파일을 읽어온다
# trim_image = im_trim(org_image)
# trim_image 변수에 결과물을 넣는다
