import cv2
import numpy as np
import math,copy
import matplotlib.pyplot as plt

cam=cv2.VideoCapture(1)

def irisrected(img, c, r):
    ll=2
    t=0
    rec=np.ones((r, 360*ll, 3), np.uint8)
    for i in range(0, 360*ll):
        i/=ll
        x=int(math.cos(math.radians(i))*r)
        y=int(math.sin(math.radians(i))*r)
        lis=get_line(c, (c[0]+x, c[1]+y))
        vert = []
        for j in range(len(lis)):
            #rec[j, t]=img[lis[j][1], lis[j][0]]
            vert+=[img[lis[j][1], lis[j][0]]]
            #cv2.circle(img, (lis[j][0], lis[j][1]), 1, (255, 0, 0))
        vert=np.reshape(vert, (1, len(lis), 3))
        vert=cv2.resize(vert, (r, 1))
        rec[:, t]=vert

        t+=1

    return rec


def iriscircle(rec, w, h, c, r):
    ll=rec.shape[1]//360
    ll2=5
    t=0
    img=cv2.bitwise_not(np.zeros((h, w, 3), np.uint8))

    for i in range(0, rec.shape[1]):
        for q in range(1, ll2+1):
            x=int(math.cos(math.radians(i/ll+(1/q)))*r)
            y=int(math.sin(math.radians(i/ll+(1/q)))*r)
            lis=get_line(c, (c[0]+x, c[1]+y))
            vert=cv2.resize(rec[:, i], (1, len(lis)))

            for j in range(len(lis)):
                #rec[j, t]=img[lis[j][1], lis[j][0]]

                img[lis[j][1], lis[j][0]]=vert[j, 0]

                #cv2.circle(img, (lis[j][0], lis[j][1]), 1, (255, 0, 0))

    return img

def get_line(start, end):

    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def closest(img, y, x, color):
    sx=1;sy=0;n=1
    t=1;ht=2;y-=1
    while(img[y, x]!=color):
        if 0<=x+sx<1088:
            x+=sx
        y+=sy
        t+=1
        if t==ht:
            n+=1
            t=0
            if sx==-1:
                sy=-1
                sx=0
            if sx==1:
                sy=1
                sx=0
            if sy==-1:
                sx=1
                sy=0
            if sy==1:
                sx=-1
                sy=0
        if n==3:
            ht+=1

    return y, x
def dfs(img, img2, j, i, width, height, Col1, Col2):
    todo = [(j, i)]
    while todo:
        j, i = todo.pop()
        if not (0 <= j < height) or not (0 <= i < width) or img[j, i] != Col1:
            continue
        img[j, i] = Col2
        img2[j, i] = Col2

        todo += [(j+1, i), (j-1, i), (j, i+1), (j, i-1)]

    return img2
def acircle(c,r):
    dots=[]
    t=3
    for i in range(360*t):
        dots+=[[int(math.cos(i/t)*r+c[0]),int(math.sin(i/t)*r+c[1])]]
    return dots

def setchart(eye, c1,r1,c2,r2):
    w=eye.shape[1]
    h=eye.shape[0]
    out=np.zeros((h,w,3),np.uint8)
    distance=get_line(c1,c2)
    for i in range(r2-r1):
        dots=acircle(distance[int(i*len(distance)/(r2-r1))],r1+i)
        for j in dots:
            movy=distance[int(i*len(distance)/(r2-r1))][1]-h//2
            movx=distance[int(i*len(distance)/(r2-r1))][0]-w//2
            out[j[1]-movy,j[0]-movx]=eye[j[1],j[0]]


    return out





ret, frame = cam.read()

h, w=frame.shape[:-1]
cenx=int(w/2)
ceny=int(h/2)
pic=False


_, chart = cv2.threshold(cv2.imread("charts/leftchart.png"), 127, 255, cv2.THRESH_BINARY)
_, chart1 =cv2.threshold(cv2.imread("charts/leftchart.png", 0), 127, 255, cv2.THRESH_BINARY)
Rnames = cv2.imread("charts/leftchartnamed.png")
gg=70
hh=60
gg1=30

while(True):
    # Capture frame-by-frame
    k=cv2.waitKey(100)
    _, frame = cam.read()
    if  k == ord(' '):
        img=frame
        cv2.imwrite('datasets/3.jpg', img)
        pic=True
        break
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 2, 500.0, 30, hh,gg,140)
    circles = np.uint16(np.around(circles))
    if len(circles) == 1:
        for circ in circles[0, :]:
            # draw the outer circle
            cv2.circle(gray2, (circ[0],circ[1]), circ[2], 255, 2)
            # draw the center of the circle
            cv2.circle(gray2, (circ[0], circ[1]), 2, 255, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = np.uint16(np.around(circles))

        cv2.imshow('1', gray2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(2):
            gray = cv2.blur(gray, (15, 15))
        _, pill = cv2.threshold(gray, gg1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(pill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            c = contours[1]

            M = cv2.moments(c)
            if M["m00"] != 0:
                PX = int(M["m10"] / M["m00"])
                PY = int(M["m01"] / M["m00"])
                Pmas = 0
                for i in range(PX, w):
                    if pill[PY, i] != 0:
                        break
                    Pmas += 1

                cv2.circle(frame, (PX, PY), Pmas, (255, 0, 0), 2)

    if  k == ord('q'):
        hh+=10
        print("hh: ",hh)
    if  k == ord('a'):
        hh-=10
        print("hh: ",hh)
    if  k == ord('w'):
        gg+=10
        print("gg: ",gg)
    if  k == ord('s'):
        gg-=10
        print("gg: ",gg)

    if  k == ord('r'):
        gg1+=2
        print("gg1: ",gg1)
    if  k == ord('f'):
        gg1-=2
        print("gg1: ",gg1)
    #Rnames=cv2.resize(Rnames, (w, h))




    #frame = cv2.bitwise_and(frame, Rnames)
    # Our operations on the frame come here

    # Display the resulting frame

    cv2.imshow('1', gray2)

    cv2.imshow('eye', frame)

    if  k == ord('z'):
        break



if not pic:
    img = cv2.imread("datasets/2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 2, 500.0, gg1, hh, gg, 140)
    circles = np.uint16(np.around(circles))
    if len(circles) == 1:
        for circ in circles[0, :]:
            pass

    for i in range(2):
        gray = cv2.blur(gray, (15, 15))
    _, pill = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(pill, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        c = contours[1]

        M = cv2.moments(c)
        if M["m00"] != 0:
            PX = int(M["m10"] / M["m00"])
            PY = int(M["m01"] / M["m00"])
            Pmas = 0
            for i in range(PX, w):
                if pill[PY, i] != 0:
                    break
                Pmas += 1
    h, w = img.shape[:-1]



cv2.imshow('many',img)
img=setchart(img, (PX, PY), Pmas, circ[:-1], circ[2])
kernel = np.ones((5,5),np.uint8)
img=cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel)
print(img.shape)


cv2.imshow('many2',img)
cv2.waitKey()
reaciris=irisrected(img, (cenx, ceny),circ[2]-2)
#circleiris=iriscircle(reaciris, w, h, (cenx, ceny), ceny-50)

##################
########################################
cv2.destroyAllWindows()
dist=35

cv2.imshow('rec',reaciris)

imgGray = cv2.cvtColor(reaciris, cv2.COLOR_BGR2GRAY)
ret, lightDots=cv2.threshold(imgGray, 200, 255, cv2.THRESH_BINARY)
_, mard=cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)

for i in range(reaciris.shape[1]):
    mianr=0
    miang=0
    mianb=0
    numb=0
    for j in range(reaciris.shape[0]):
        if mard[j, i]==255 and lightDots[j,i]==0:
            mianr+=reaciris[j, i][2]
            miang+=reaciris[j, i][1]
            mianb+=reaciris[j, i][0]
            numb+=1
    mianr/=numb
    miang/=numb
    mianb/=numb
    for j in range(reaciris.shape[0]):
        if mard[j, i]==255 :
            if (mianr-dist<=reaciris[j, i][2] and miang-dist<=reaciris[j, i][1] and mianb-dist<=reaciris[j, i][0]) or lightDots[j,i]==255:
                reaciris[j, i]=(255,255,255)

print(reaciris.shape)
_, imgGray=cv2.threshold(cv2.cvtColor(reaciris,cv2.COLOR_BGR2GRAY), 249, 255, cv2.THRESH_BINARY)


imgGray=cv2.morphologyEx(imgGray, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
cv2.imshow('21',imgGray)
cv2.waitKey()
bugs=dfs(imgGray, imgGray, 1, 1, reaciris.shape[1], reaciris.shape[0], 0, 255)

bugs=iriscircle(bugs, w, h, (cenx, ceny), circ[2]-5)
bugs=cv2.cvtColor(bugs, cv2.COLOR_BGR2GRAY)




#bugs, bugs2 = setchart(bugs2, bugs, nx, [0, 215, 434, 653, 877, 1088], ny, [0, 82, 308, 510, 734, 816])
bugs=cv2.resize(bugs, (int(w*1.7), int(h*1.7)))
bugs=cv2.morphologyEx(bugs, cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
_,bugs=cv2.threshold(bugs,250,255,cv2.THRESH_BINARY)
#########################################
##################

contours, hierarchy = cv2.findContours(bugs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#bug2=copy.deepcopy(bugs)



chart2=copy.deepcopy(chart1)
for c in range(1, len(contours)):
    c=contours[c]
    M = cv2.moments(c)
    if M["m00"]!=0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if chart1[cY, cX]==0:
            cY, cX=closest(chart2, cY, cX, 255)
        chart1 = dfs(chart1, chart1, cY, cX, chart1.shape[1], chart1.shape[0], 255, 127)

#charted=cv2.bitwise_and(chart1, bugs)

charted=cv2.cvtColor(chart1, cv2.COLOR_GRAY2BGR)
notneededvar=200

for i in range(charted.shape[1]):
    for j in range(charted.shape[0]):

        if chart1[j, i]==127:
            charted[j, i]=[Rnames[j, i][0]//1.5, Rnames[j, i][1]//1.5, Rnames[j, i][2]//1.5]
        if bugs[j, i]==0 and 200<Rnames[j, i][0]:
            charted[j, i]=[100, 100, 100]


cv2.imshow('problems', bugs)
cv2.imshow('rect', reaciris)
cv2.imshow('chart', charted)
cv2.waitKey()
print(1)
cv2.waitKey()
cv2.destroyAllWindows()