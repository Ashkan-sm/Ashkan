import cv2
import numpy as np
import copy,math
cam=cv2.VideoCapture(1)
print(1)
def irisrected(img,c,r):
    ll=2
    t=0
    rec=np.ones((r,360*ll,3),np.uint8)
    for i in range(0,360*ll):
        i/=ll
        x=int(math.cos(math.radians(i))*r)
        y=int(math.sin(math.radians(i))*r)
        lis=get_line(c,(c[0]+x,c[1]+y))
        vert = []
        for j in range(len(lis)):
            #rec[j,t]=img[lis[j][1],lis[j][0]]
            vert+=[img[lis[j][1],lis[j][0]]]
            #cv2.circle(img,(lis[j][0],lis[j][1]),1,(255,0,0))
        vert=np.reshape(vert,(1,j+1,3))
        vert=cv2.resize(vert,(r,1))
        rec[:, t]=vert

        t+=1

    cv2.imshow('1',rec)
    cv2.imshow('2',img)
    cv2.waitKey()
    return rec


def iriscircle(rec,w,h,c,r):
    ll=rec.shape[1]//360
    ll2=5
    t=0
    img=np.zeros((h,w,3),np.uint8)

    for i in range(0,rec.shape[1]):
        for q in range(1,ll2+1):
            x=int(math.cos(math.radians(i/ll+(1/q)))*r)
            y=int(math.sin(math.radians(i/ll+(1/q)))*r)
            lis=get_line(c,(c[0]+x,c[1]+y))
            vert=cv2.resize(rec[:,i],(1,len(lis)))

            for j in range(len(lis)):
                #rec[j,t]=img[lis[j][1],lis[j][0]]

                img[lis[j][1],lis[j][0]]=vert[j,0]

                #cv2.circle(img,(lis[j][0],lis[j][1]),1,(255,0,0))

    cv2.imshow('1',rec)
    cv2.imshow('2',img)
    cv2.waitKey()
    return rec

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

def closest(img,y,x,color):
    sx=1;sy=0;n=1
    t=1;ht=2;y-=1
    while(img[y,x]!=color):
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

    return y,x
def dfs(img,img2,j,i,width,height,Col1,Col2):
    todo = [(j,i)]
    while todo:
        j,i = todo.pop()
        if not (0 <= j < height) or not (0 <= i < width) or img[j,i] != Col1:
            continue
        img[j,i] = Col2
        img2[j,i] = Col2

        todo += [(j+1,i), (j-1,i), (j,i+1), (j,i-1)]

    return img2

def setchart(chart,names,p1x,p2x,p1y,p2y):
    h,w=chart.shape[:2]
    t2=(h/2-70)**2
    output=np.zeros((h,w),np.uint8)
    output2=np.zeros((h,w,3),np.uint8)
    for j in range(h):
        ir1=int(abs(t2-(h//2-j)**2)**0.5)
        p2x[1]=w//2-ir1
        p2x[4]=w//2+ir1
        p1x[1]=w//2-ir1
        p1x[4]=w//2+ir1
        if p2x[0]<p2x[1]<p2x[2]<p2x[3]<p2x[4]<p2x[5] and p1x[0]<p1x[1]<p1x[2]<p1x[3]<p1x[4]<p1x[5]:
            for i in range(len(p1x)-1):
                output[j:j+1,p1x[i]:p1x[i+1]]=cv2.resize(chart[j:j+1,p2x[i]:p2x[i+1]],(p1x[i+1]-p1x[i],1))
                output2[j:j+1,p1x[i]:p1x[i+1]]=cv2.resize(names[j:j+1,p2x[i]:p2x[i+1]],(p1x[i+1]-p1x[i],1))
        else:
            output[j:j+1,:]=chart[j:j+1,:]
            output2[j:j+1,:]=names[j:j+1,:]


    for j in range(w):
        ir1=int(abs(t2-(w//2-j)**2)**0.5)
        p2y[1]=h//2-ir1
        p2y[4]=h//2+ir1
        p1y[1]=h//2-ir1
        p1y[4]=h//2+ir1
        if p2y[0]<p2y[1]<p2y[2]<p2y[3]<p2y[4]<p2y[5] and p1y[0]<p1y[1]<p1y[2]<p1y[3]<p1y[4]<p1y[5]:
            for i in range(len(p1y)-1):
                output[p1y[i]:p1y[i+1],j:j+1]=cv2.resize(output[p2y[i]:p2y[i+1],j:j+1],(1,p1y[i+1]-p1y[i]))
                output2[p1y[i]:p1y[i+1],j:j+1]=cv2.resize(output2[p2y[i]:p2y[i+1],j:j+1],(1,p1y[i+1]-p1y[i]))
        else:
            output[:,j:j+1]=chart[:,j:j+1]
            output2[:,j:j+1]=names[:,j:j+1]

    return output,output2

ret, frame = cam.read()

h,w=frame.shape[:-1]
cenx=int(w/2)
ceny=int(h/2)
pic=False


ret, chart = cv2.threshold(cv2.imread("charts/leftchart.png"),127,255,cv2.THRESH_BINARY)
ret,chart1 =cv2.threshold(cv2.imread("charts/leftchart.png",0),127,255,cv2.THRESH_BINARY)
Rnames = cv2.imread("charts/leftchartnamed.png")


cv2.imshow('1',chart1[100:,:])
cv2.waitKey()
while(True):
    # Capture frame-by-frame
    k=cv2.waitKey(1)
    ret, frame = cam.read()
    if  k == ord(' '):
        img=frame
        cv2.imwrite('datasets/3.jpg',img)
        pic=True
        break
    #Rnames=cv2.resize(Rnames,(w,h))

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for i in range(2):
        gray = cv2.blur(gray, (15, 15))
    _,pill=cv2.threshold(gray,25,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(pill,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)>1:
        c = contours[1]

        M = cv2.moments(c)
        if M["m00"]!=0:
            PX = int(M["m10"] / M["m00"])
            PY = int(M["m01"] / M["m00"])
            Pmas=0
            for i in range(PX,w):
                if pill[PY,i]!=0:
                    break
                Pmas+=1
            if 0< int((cenx - int(h / 2) + 50) * 1.7)< int((PX - Pmas) * 1.7)<int((PX + Pmas) * 1.7)< int((cenx + int(h / 2) - 50) * 1.7)<1088:
                if 0<int((ceny - int(h / 2) + 50) * 1.7)< int((PY - Pmas) * 1.7)<int((PY + Pmas) * 1.7)< int((ceny + int(h / 2) - 50) * 1.7)<816:

                    nx=[0, int((cenx - int(h / 2) + 50) * 1.7), int((PX - Pmas) * 1.7),int((PX + Pmas) * 1.7), int((cenx + int(h / 2) - 50) * 1.7),1088]
                    ny=[0, int((ceny - int(h / 2) + 50) * 1.7), int((PY - Pmas) * 1.7),
                    int((PY + Pmas) * 1.7), int((ceny + int(h / 2) - 50) * 1.7),816]

                    chart2, Rnames2 = setchart(chart1, Rnames, nx , [0, 215, 434, 653, 877, 1088],ny, [0, 82, 308, 510, 734, 816])

                    chart2=cv2.cvtColor(chart2,cv2.COLOR_GRAY2BGR)
                    chart2=cv2.resize(chart2,(w,h))

                    frame=cv2.bitwise_and(frame,chart2)
            cv2.circle(frame,(PX,PY),Pmas,(255,0,0),2)

    cv2.circle(frame,(cenx,ceny),int(h/2)-50,(255,0,0),2)
    #frame = cv2.bitwise_and(frame, Rnames)
    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('eye',frame)
    cv2.imshow('eye1',pill)
    if  k == ord('q'):
        break
cv2.destroyAllWindows()
if not pic:
    img = cv2.imread("datasets/2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
cv2.waitKey()

reaciris=irisrected(img,(cenx,ceny),ceny-50)
circleiris=iriscircle(reaciris,w,h,(cenx,ceny),ceny-50)



imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


ret,lightDots=cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY)


##################
########################################
for i in range(1):
    imgGray=cv2.blur(imgGray,(5,5))

ekh=48
bugs=np.zeros((h,w),np.uint8)
for y in range(h):
    mi=0
    t=1
    for x in range(w):
        if (cenx-x)**2+(ceny-y)**2<(ceny-50)**2:
            mi+=imgGray[y,x]
            t+=1
    mi/=t
    for x in range(w):
        if (cenx - x) ** 2 + (ceny - y) ** 2 < (ceny - 50) ** 2:
            if mi-imgGray[y,x]>ekh:
                bugs[y, x] = 255
    win=copy.deepcopy(bugs)
    cv2.line(win,(0,y),(w,y),127,3)

    cv2.imshow('2',imgGray)
    cv2.imshow('1',win)
    #cv2.waitKey(1)

for x in range(w):
    mi=0
    t=1
    for y in range(h):
        if (cenx-x)**2+(ceny-y)**2<(ceny-50)**2:
            mi+=imgGray[y,x]
            t+=1
    mi/=t
    for y in range(h):
        if (cenx - x) ** 2 + (ceny - y) ** 2 < (ceny - 50) ** 2:
            if mi-imgGray[y,x]>ekh:
                bugs[y, x] = 255
    win=copy.deepcopy(bugs)
    cv2.line(win,(x,0),(x,h),127,3)

    cv2.imshow('2',imgGray)
    cv2.imshow('1',win)
    #cv2.waitKey(1)

bugs=cv2.bitwise_not(bugs)
dfs(bugs,bugs,PY,PX,w,h,0,255)
bugs=cv2.resize(bugs,(int(w*1.7),int(h*1.7)))

#########################################
##################

contours, hierarchy = cv2.findContours(bugs,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


#bug2=copy.deepcopy(bugs)

nx = [0, int((cenx - int(h / 2) + 50) * 1.7), int((PX - Pmas) * 1.7), int((PX + Pmas) * 1.7),
      int((cenx + int(h / 2) - 50) * 1.7), 1088]
ny = [0, int((ceny - int(h / 2) + 50) * 1.7), int((PY - Pmas) * 1.7),
      int((PY + Pmas) * 1.7), int((ceny + int(h / 2) - 50) * 1.7), 816]

chart1, Rnames = setchart(chart1, Rnames, nx, [0, 215, 434, 653, 877, 1088], ny, [0, 82, 308, 510, 734, 816])
cv2.imshow('1',chart1)
cv2.waitKey()
cv2.destroyAllWindows()

for c in range(1,len(contours)):
    c=contours[c]

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if chart1[cY,cX]==0:
        cY,cX=closest(chart1,cY,cX,255)
    chart1 = dfs(chart1, chart1, cY, cX, chart1.shape[1], chart1.shape[0],255,127)

#charted=cv2.bitwise_and(chart1,bugs)

charted=cv2.cvtColor(chart1,cv2.COLOR_GRAY2BGR)
notneededvar=200
for i in range(charted.shape[1]):
    for j in range(charted.shape[0]):

        if chart1[j,i]==127:
            charted[j,i]=[Rnames[j,i][0]//1.5,Rnames[j,i][1]//1.5,Rnames[j,i][2]//1.5]
        if bugs[j,i]==0 and 200<Rnames[j,i][0]:
            charted[j,i]=[100,100,100]


cv2.destroyAllWindows()
cv2.imshow('problems',bugs)
cv2.imshow('chart',charted)
cv2.imshow('eye',img)
cv2.waitKey()
cv2.destroyAllWindows()
