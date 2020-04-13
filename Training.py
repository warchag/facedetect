import os,cv2
from pil import Image
import numpy as np

def train_classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces=[]
    ids=[]
    for imgs in path:
        img=Image.open(imgs).convert("L")
        imgeNp =np.array(img,'uint8')
        id = int(os.path.split(imgs)[1].split(".")[1])
        faces.append(imgeNp)
        ids.append(id)
  
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("aomclassifier.xml")
train_classifier("data")        