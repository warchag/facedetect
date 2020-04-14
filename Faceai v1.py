import cv2
Facecase = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def draw_boundary(img,classifier,scalFactor,minNeigh,color,clf,train):
    myimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(myimg,scalFactor,minNeigh)
    coords=[]
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        if train!="train":
            id,con= clf.predict(myimg[y:y+h,x:x+w])
            print(f"{con}%".format(round(100-con)))
            if con <= 50 :
                cv2.putText(img,"aom ka",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
            else:
                cv2.putText(img,"unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords = [x,y,w,h]
    return img,coords
def detect(img,facecase,mem_id,clf,train):
    img,coordsFace= draw_boundary(img,facecase,1.1,12,(0,0,255),clf,train)
    if len(coordsFace) == 4:
        result = img[coordsFace[1]:coordsFace[1]+coordsFace[3],coordsFace[0]:coordsFace[0]+coordsFace[2]]
        if train == "train":
            create_dataFace(result,1,mem_id)
    return img,coordsFace

def create_dataFace(img,id,img_id):
    cv2.imwrite(f"data/pic.{id}.{img_id}.jpg",img)



if __name__ =="__main__":
    mem_id = 1
    cap = cv2.VideoCapture("123.mp4")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("aomclassifier.xml")
     
    
    while True:
        mem_id+=1
        ret,frame =cap.read()
        frame,cord = detect(frame,Facecase,mem_id,clf,"xtrain")
        cv2.imshow('myvideo',frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cap.release()
    cap.destroyAllwindows()    