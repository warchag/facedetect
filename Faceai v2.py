import cv2
Facecase = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
tracker = cv2.TrackerMedianFlow_create()
ontracker = False
def draw_boundary(img,classifier,scalFactor,minNeigh):
    global ontracker
    myimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = Facecase.detectMultiScale(myimg,scalFactor,minNeigh)
    coords=[]
    if not ontracker:
        for(x,y,w,h) in features:
            coords = [x,y,w,h]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            if tracker.init(img,(x,y,w,h)):
                ontracker = True
    else:
        ok,bbox = tracker.update(img)
        if ok:
            p1 = (int(bbox[0]),int(bbox[1])) 
            p2 = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))  
            cv2.rectangle(img,p1,p2,(0,255,0),2)         
    return img,coords


if __name__ =="__main__":
    cap = cv2.VideoCapture("2.mp4")
    while True:
        x,frame =cap.read()
        frame,coord = draw_boundary(frame,Facecase,1.3,5)
        cv2.imshow('myvideo',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cap.destroyAllwindows()    