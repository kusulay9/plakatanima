import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

resim_adresler = os.listdir("veriseti")

img = cv2.imread("veriseti/"+resim_adresler[6])
img = cv2.resize(img,(500,500))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

img_bgr = img
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ir_img = cv2.medianBlur(img_gray,5)
ir_img = cv2.medianBlur(img_gray,5)

plt.imshow(ir_img,cmap="gray")
plt.show()

medyan = np.median(ir_img)
low = 0.67*medyan
high = 1.33*medyan
kenarlik = cv2.Canny(ir_img,low,high)

plt.imshow(kenarlik,cmap="gray")
plt.show()

kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)

plt.imshow(kenarlik,cmap="gray")
plt.show()

cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt,key=cv2.contourArea,reverse=True)

H,W = 500,500
plaka = None

for c in cnt:
    rect = cv2.minAreaRect(c)
    (x,y),(w,h),r = rect
    if(w>h and w>h*2) or (h>w and h>w*2):
       box = cv2.boxPoints(rect)
       box = np.int64(box)

       minx = np.min(box[:,0])
       miny = np.min(box[:,1])
       maxx = np.max(box[:,0])
       maxy = np.max(box[:,1])

       muh_plaka = img_gray[miny:maxy,minx:maxx].copy()
       muh_medyan = np.median(muh_plaka)


       kon1 = muh_medyan>100 and muh_medyan<200
       kon2 = h>80 and w<200
       kon3 = w>80 and h<200

       print(f"muh_plaka medyan{muh_medyan} genişlik:{w} yükseklik:{h}")
       plt.figure()
       kon = False
       if(kon1 and (kon2 or kon3)):
          cv2.drawContours(img,[box],0,(0,255,0),2)
          plaka = [minx,miny,w,h]

        
          plt.title("Plaka Tespit Edildi!!!")
          kon = True
       else:
            cv2.drawContours(img,[box],0,(0,0,255),2)
            plt.title("Plaka Tespit Edilemedi!!!")
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            plt.show()
            if(kon):
                break
def findListOfMatchingChars(possibleChar, listOfChars):
           
    listOfMatchingChars = []                

    for possibleMatchingChar in listOfChars:                
        if possibleMatchingChar == possibleChar:    
            continue                                
                    
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

               
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        
    
    return listOfMatchingChars             


def getOtsuThreshold(im):
    size = 256 
    buckets = np.zeros([size]) 
    imy = im.shape[0]
    imx = im.shape[1]
    image_size = imx*imy
    for i in xrange(imy):
        for j in xrange(imx):
            buckets[im[i][j]] += 1
        


