import cv2 as cv 
import numpy as np
import time
import tensorflow as tf
import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN
from sklearn.neighbors import KNeighborsClassifier


from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


lable = ['unmasked','masked']
names = ['Aon','Most','First']
def togray(frame):
    return cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

def process_img(frame):
    result = cv.resize(frame,(224,224))
    result = result.astype('float64')
    result = np.expand_dims(result,axis=0)       
    result = utils.preprocess_input(result,version=2)
    return result

class setupmodel:
    def __init__(self,path,facedetect,model):
        self.Fcount = 0
        self.data  = []
        self.label = []
        self.Fname = "Data"
        for base, dirs, files in os.walk(path):
            for Files in files:
                self.Fcount += 1
        Lcount = 0;
        Mlabel = 0;
        for i in range(1,self.Fcount+1):
            img_name = path+"/"+self.Fname+str(i)+".jpg"
            img = cv.imread(img_name)
            result = facedetect.detect_faces(img)
            for j in result:
                confi = j['confidence']
                if confi > 0.95:
                    box = j['box']
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    face = img[y:y+h,x:x+w]
                    data = model.predict(process_img(face))
                    dshape = data.shape
                    data = np.reshape(data,(dshape[1]*dshape[2]*dshape[3]))
                    self.data.append(data)
            Lcount += 1
            self.label.append(Mlabel)
            if(Lcount % 3 == 0):
                Mlabel+=1

        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(self.data,self.label)
    def __call__(self,frame,model):
        data = model.predict(process_img(frame=frame))
        dshape = data.shape
        data = np.reshape(data,(dshape[1]*dshape[2]*dshape[3]))
        data = np.expand_dims(data,axis=0)
        return self.model.predict(data)

class facedetect:
    def __init__(self):
        self.model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    def __call__(self,frame):
        gray = togray(frame = frame)
        return (self.model.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        flags=cv.CASCADE_SCALE_IMAGE))
        


class maskrec:
    def __init__(self):
        self.model = tf.keras.models.load_model('saved_model/my_model')
    def __call__(self,frame):
        # G_frame = togray(frame=frame)
        G_frame = frame
        G_frame = cv.resize(G_frame, (128,128))
        G_frame = G_frame/255.0
        # G_frame = G_frame.reshape(G_frame.shape[0],G_frame.shape[1],1)
        G_frame = np.expand_dims(G_frame,axis=0)
        result = self.model.predict(G_frame)
        if result[0][0] >= 0.99 :
            return 1
        else:
            return 0


facedetect1 = facedetect()
facedetect2 = MTCNN()
maskrec1 = maskrec()
model = VGGFace(model='resnet50',include_top=False)
model1 = setupmodel(path="D:/ML/Facerect/Data1",facedetect=facedetect2,model=model)

cap = cv.VideoCapture(0)
pTime = 0
Fcount = 0
CFrame = 1
while True:
    sccess, frame = cap.read()
    result = facedetect1(frame)
    if (len(result) >0):
        for i in result:
            x,y,w,h = i
            face = frame[y:y+h,x:x+w]
            mask = maskrec1(face)
            if mask == 1:
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,155,255),2)
                cv.putText(frame, f'{lable[mask]}',(x,y-40),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
            else:
                name = model1(frame=face,model = model)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,155,255),2)
                cv.putText(frame, f'{lable[mask]}',(x,y-40),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                cv.putText(frame, f'{names[name[0]]}',(x,y-10),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, f'Fps: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,
                3,(0,255,0),2)
    cv.imshow('Python Window', frame)
    Fcount+=1
    if cv.waitKey(10) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break