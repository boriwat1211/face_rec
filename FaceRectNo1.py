
import cv2 as cv 
import numpy as np
from facenet_pytorch import MTCNN,InceptionResnetV1
import time
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import local_binary_pattern
import os
import torch



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resize_img(img):
    try:
        i_width = 160
        i_height = 160
        return cv.resize(img,(i_width,i_height))
    except cv.error as a:
        print("Invalid frame")



def LBP(img):
    radius  = 2
    numPoints = 8 * radius
    gray_img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_img,numPoints,radius, method="default")
    hist, _ = np.histogram(lbp, bins=np.arange(2**8+1))
    return hist


class custom_mtcnn:
    def __init__(self,resize, *args, **kwargs):
        self.resize = resize
        self.mtcnn = MTCNN(*args,**kwargs)
    def __call__(self,frame):
        height , width  , rgb= frame.shape
        if self.resize <= 1:
            frame = cv.resize(frame,(int(width*self.resize),int(height*self.resize)),interpolation = cv.INTER_AREA)
        face = self.mtcnn.detect([frame])
        tensor = self.mtcnn(frame)
        return face,tensor;



class setup_model1:
    def __init__(self,path,mtcnn,resnet):
        self.Fcount = 0
        self.data  = []
        self.label = []
        self.Fname = "Data"
        for base, dirs, files in os.walk(path):
            for Files in files:
                self.Fcount += 1
        Lcount = 0;
        label = 0;
        for i in range(1,self.Fcount+1):
            img_name = path+"/"+self.Fname+str(i)+".jpg"
            img = cv.imread(img_name)
            rgb_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            result ,tensor= mtcnn(rgb_img)
            x,y,width,height = int(result[0][0][0][0]/mtcnn.resize),int(result[0][0][0][1]/mtcnn.resize),int(result[0][0][0][2]/mtcnn.resize),int(result[0][0][0][3]/mtcnn.resize)
            crop_img = rgb_img[y:height,x:width]
            resize_crop_img = resize_img(crop_img)
            self.data.append(LBP(resize_crop_img))
            Lcount += 1
            self.label.append(label)
            if(Lcount % 3 == 0):
                label+=1
    def __call__(self,frame,tensor,x,y,width,height):
        if(x>=0 and y>=0 and width>=0 and height>=0):
            crop_face = frame[y:height,x:width]
            resize_crop_face = resize_img(img = crop_face)
            return(LBP(resize_crop_face))   
        else:
            return None
class setup_model2:
    def __init__(self,path,mtcnn,resnet):
        self.Fcount = 0
        self.data  = []
        self.label = []
        self.Fname = "Data"
        for base, dirs, files in os.walk(path):
            for Files in files:
                self.Fcount += 1
        Lcount = 0;
        label = 0;
        for i in range(1,self.Fcount+1):
            img_name = path+"/"+self.Fname+str(i)+".jpg"
            img = cv.imread(img_name)
            rgb_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            result ,tensor= mtcnn(rgb_img)
            tensor = tensor.cuda().float()     
            self.data.append(resnet(tensor).cpu().detach().numpy()[0])
            Lcount += 1
            self.label.append(label)
            if(Lcount % 3 == 0):
                label+=1
    def __call__(self,frame,tensor,x,y,width,height):
        if(x>=0 and y>=0 and width>=0 and height>=0):
            tensor = tensor.cuda().float() 
            result =  resnet(tensor).cpu().detach().numpy()
            return result[0]
        else:
            return None       
class KNN:
    def __init__(self,data):
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(data.data,data.label)
    def __call__(self,data):
        if(data is not None):
            result = self.model.predict([data])
            return result
        else :
            return [3]

class yolov5:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./work01.pt')
        self.model.conf = 0.20  
        self.model.iou = 0.45 
        self.model.classes = None
    def __call__(self,frame):
        print(frame.shape)
        results = self.model(frame)
        return (results.pandas().xyxy[0].to_numpy()) 

resize = 1.0
resnet = InceptionResnetV1(pretrained='vggface2',classify=False,device=device).eval()
mtcnn = custom_mtcnn(resize=resize,factor=0.5,keep_all=True,device=device)
data = setup_model2(path="D:/ML/Facerect/Data1",mtcnn = mtcnn,resnet = resnet)
knn =  KNN(data=data)
model = yolov5()
cap = cv.VideoCapture(0)
pTime = 0
Fcount = 0
CFrame = 1
name = ['aon','most','first','error']
while True:
    sccess, frame = cap.read()
    if(Fcount%CFrame == 0):
        face = model(frame=frame)
        for i in range (len(face)):
            if face[i][5] == 1  :
                x,y,w,h = int(face[i][0]*0.9),int(face[i][1]*0.8),int(face[i][2]*1.1),int(face[i][3])
                no_mask = frame[y:h,x:w]
                cv.rectangle(frame,(x,y),(w,h),(255,155,255),2)
                result , tensor = mtcnn(frame=no_mask)
                if not(result[0][0] is None):
                    faces = result[0][0]
                    scores = result[1][0]
                    for j in  range(len(faces)):
                        tensor_s = tensor[j]
                        CF_x,CF_y,CF_w,CF_h = int((faces[j][0]/resize)),int((faces[j][1]/resize)),int((faces[j][2]/resize)),int((faces[j][3]/resize))
                        cv.putText(frame, f'{name[knn(data(frame=no_mask[CF_y:CF_h,CF_x:CF_w],tensor=tensor_s.unsqueeze_(0),x=CF_x,y=CF_y,width=CF_w,height=CF_h))[0]]}',(x,y-50),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                        cv.putText(frame, f'{face[i][6]}',(x,y-20),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, f'Fps: {int(fps)}',(20,70),cv.FONT_HERSHEY_PLAIN,
                3,(0,255,0),2)
    cv.imshow('Python Window', frame)
    Fcount+=1
    if cv.waitKey(25) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break