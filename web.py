from re import I
import streamlit as st
import numpy as np
from PIL import Image 
import cv2 as cv
import torch
from facenet_pytorch import InceptionResnetV1,MTCNN
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './yolov5')
import os
import time
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import torch.backends.cudnn as cudnn
from torchvision import transforms, models

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import pyrebase
device = 'cuda' if torch.cuda.is_available() else 'cpu'
firebaseConfig = {
  "apiKey": "AIzaSyBXRqz3CKO8bWY0K-IB_3hFuvGGPZe0OE8",
  "authDomain": "mlproject-57cf7.firebaseapp.com",
  "databaseURL": "https://mlproject-57cf7-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "mlproject-57cf7",
  "storageBucket": "mlproject-57cf7.appspot.com",
  "messagingSenderId": "1047664940326",
  "appId": "1:1047664940326:web:f6b244c01094f8fa261de7",
  "measurementId": "G-H7M9ZVZZ3Q",
  "serviceAccount": "serviceAccount.json"
}
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()
class face_list:
    def __init__(self):
        self.list_face=[]
        self.max_face=20
    def insert(self,pic_face,id_face,name_face):
        data = {
            "pic":pic_face,
            "id":id_face,
            "name":name_face
        }
        check = self.check(id_face=id_face,name_face=name_face)
        if len(self.list_face) < self.max_face:
            if check:
                self.list_face.append(data)
            else:
                self.update(id_face=id_face,name_face=name_face,pic_face=pic_face)
        else:
            if check:
                del self.list_face[0]
                self.list_face.append(data)
            else:
                self.update(id_face=id_face,name_face=name_face,pic_face=pic_face)
    def update(self,id_face,name_face,pic_face):
        current = len(self.list_face)-1
        while current >=0:
            if(self.list_face[current]['id']==id_face):
                if(self.list_face[current]['name']=='Unknow' and name_face != 'Unknow'):
                    self.list_face[current]['name'] = name_face
                    self.list_face[current]['pic'] = pic_face
            current-=1
    def check(self,id_face,name_face):
        current = len(self.list_face)-1
        check = True
        while current >=0:
            if(self.list_face[current]['id']==id_face):
                check = False
            current-=1
        return check
    def clear(self):
        self.list_face.clear()
class face_data:
    def __init__(self):
        self.facecount =0
        self.face_data = db.child("Data").get()
        self.face_data = self.face_data.val()
        self.face_label = db.child("Label").get()
        self.face_label = self.face_label.val()
        self.np_face_data = []
        if(self.face_data is not None):
            for i in range (len(self.face_data)):
                self.facecount+=1
                self.face_data[i] = self.face_data[i].replace("[",'')
                self.face_data[i] = self.face_data[i].replace("]",'')
                self.np_face_data.append(np.fromstring(self.face_data[i],dtype=float,sep=','))
        

    def upload_label(self,face,name):
        resnet = InceptionResnetV1(pretrained='vggface2',classify=False,device=device).eval()
        face = cv.resize(face,(160,160))
        face = np.expand_dims(face,axis=0)
        face = torch.tensor(face/255)
        face = face.permute(0,3,1,2)
        face = face.float().to(device)
        result = resnet(face).cpu().detach().numpy()
        up_face_data = {str(self.facecount):np.array2string(result[0],separator=',')}
        up_face_label = {str(self.facecount):name}
        db.child("Data").update(up_face_data)
        db.child("Label").update(up_face_label)
        self.updata_label()
        return "Uploaded"
    
    def updata_label(self):
        self.facecount = 0
        self.facecount =0
        self.face_data = db.child("Label").get()
        self.face_data = self.face_data.val()
        if(self.face_data is not None):
            for i in self.face_data:
                self.facecount+=1

    def __call__(self,face):
        list_result = []
        for L_data in self.np_face_data:
            result = np.dot(face, L_data)/(np.linalg.norm(face) *
                                          np.linalg.norm(L_data))
            list_result.append(result)
        np_result = np.asarray(list_result)
        l_result = np.argmax(list_result)
        if np_result[l_result] > 0.7:
            return self.face_label[l_result]
        else :
            return "Unknow"


def face_detect(image):
    mtcnn = MTCNN(factor=0.7,keep_all=True,device=device)
    cv_image = np.array(image.convert('RGB'))
    cv_image = cv.cvtColor(cv_image,1)
    face , tensor= mtcnn.detect([cv_image])    
    return face[0],cv_image

def main():
    face_data_man = face_data()
    st.title("Face Mask Detection")
    main_option = st.selectbox(
        'Select work',
        ('Detection', 'Upload Photo'))
    st.write('You selected:', main_option)
    if main_option == 'Detection':
        detec_sub_option = st.selectbox(
            'Select input',
            ('Video', 'Webcam'))
        if detec_sub_option == 'Webcam':
            mask_model = torch.load("./torch_mask7.pt")
            mask_model.eval()
            resnet = InceptionResnetV1(pretrained='vggface2',classify=False,device='cuda').eval()
            facelist = face_list()
            

            openwebcam = st.checkbox('Open webcam')
            if openwebcam :
                Frame_video = st.image([])
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    face_text1 = st.empty()
                    face_img1 = st.image([])
                    face_text6 = st.empty()
                    face_img6 = st.image([])
                    face_text11 = st.empty()
                    face_img11 = st.image([])
                    face_text16 = st.empty()
                    face_img16 = st.image([])
                with col2:
                    face_text2 = st.empty()
                    face_img2 = st.image([])
                    face_text7 = st.empty()
                    face_img7 = st.image([])
                    face_text12 = st.empty()
                    face_img12 = st.image([])
                    face_text17 = st.empty()
                    face_img17 = st.image([])
                with col3:
                    face_text3 = st.empty()
                    face_img3 = st.image([])
                    face_text8 = st.empty()
                    face_img8 = st.image([])
                    face_text13 = st.empty()
                    face_img13 = st.image([])
                    face_text18 = st.empty()
                    face_img18 = st.image([])
                with col4:
                    face_text4 = st.empty()
                    face_img4 = st.image([])
                    face_text9 = st.empty()
                    face_img9 = st.image([])
                    face_text14 = st.empty()
                    face_img14 = st.image([])
                    face_text19 = st.empty()
                    face_img19 = st.image([])
                with col5:
                    face_text5 = st.empty()
                    face_img5 = st.image([])
                    face_text10 = st.empty()
                    face_img10 = st.image([])
                    face_text15 = st.empty()
                    face_img15 = st.image([])
                    face_text20 = st.empty()
                    face_img20 = st.image([])
                cfg = get_config()
                cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
                attempt_download("deep_sort_pytorch/configs/deep_sort.yaml", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
                deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
                device = select_device('cuda')
                half = device.type != 'cpu'
                model = attempt_load("crowdhuman_yolov5m.pt", map_location=device)  
                stride = int(model.stride.max())  
                imgsz = check_img_size(640, s=stride)  
                names = model.module.names if hasattr(model, 'module') else model.names
                model.half()
                cudnn.benchmark = True  
                dataset = LoadStreams('0', img_size=imgsz, stride=stride)
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
                t0 = time.time()
                for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    t1 = time_sync()
                    pred = model(img, augment= False)[0]
                    pred = non_max_suppression(
                        pred,0.8,0.5, classes=1, agnostic=False)
                    t2 = time_sync()

                    for i, det in enumerate(pred):
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                        s += '%gx%g ' % img.shape[2:]

                        annotator = Annotator(im0, line_width=2, pil=not ascii)
                        if det is not None and len(det):
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]

                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            if len(outputs) > 0:
                                for j, (output, conf) in enumerate(zip(outputs, confs)): 
                                
                                    bboxes = output[0:4]
                                    id = output[4]
                                    cls = output[5]
                                    c = int(cls)  # integer class
                
                                                
                                    x,y,w,h=bboxes
                                    normal = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    # current_frame = annotator.result()
                                    current_frame = im0
                                    face = current_frame[y:h,x:w]
                                    face = cv.resize(face,(224,224))
                                    face = cv.cvtColor(face,cv.COLOR_BGR2RGB)
                                    face = np.expand_dims(face,axis=0)
                                    face = torch.tensor(face/255)
                                    face = face.permute(0,3,1,2)
                                    face = normal(face)
                                    face = face.float().to(device)
                                    result = mask_model(face)
                                    if result[0][0] > 0.70:
                                        # label = f'{id} {"masked"}'
                                        annotator.box_label(bboxes, color=(0,255,0))
                                    else:
                                        face2 = current_frame[y:h,x:w]
                                        face2 = cv.cvtColor(face2,cv.COLOR_BGR2RGB)
                                        face2 = cv.resize(face2,(160,160))
                                        face_save = face2
                                        face2 = np.expand_dims(face2,axis=0)
                                        face2 = torch.tensor(face2/255)
                                        face2 = face2.permute(0,3,1,2)
                                        face2 = face2.float().to(device)
                                        result2 = resnet(face2).cpu().detach().numpy()
                                        nameface = face_data_man(face = result2[0])
                                        label = f'{id} {"Nomask"} {nameface}'
                                        # facelist.insert(pic_face=face_save,id_face=id,name_face=nameface)
                                        annotator.box_label(bboxes, color=(0,0,255))
                        else:
                            deepsort.increment_ages()
                        im0 = annotator.result()
                        # im0 = cv.resize(im0,(1280,960))
                        im0 = cv.cvtColor(im0,cv.COLOR_BGR2RGB)
                        Frame_video.image(im0)
                        if len(facelist.list_face)>0:
                            if len(facelist.list_face)>=1:
                               face_text1.text(str(facelist.list_face[0]['id'])+" "+str(facelist.list_face[0]['name']))
                               face_img1.image(facelist.list_face[0]['pic'])

                            if len(facelist.list_face)>=2:
                               face_text2.text(str(facelist.list_face[1]['id'])+" "+str(facelist.list_face[1]['name']))
                               face_img2.image(facelist.list_face[1]['pic'])

                            if len(facelist.list_face)>=3:
                               face_text3.text(str(facelist.list_face[2]['id'])+" "+str(facelist.list_face[2]['name']))
                               face_img3.image(facelist.list_face[2]['pic'])

                            if len(facelist.list_face)>=4:
                               face_text4.text(str(facelist.list_face[3]['id'])+" "+str(facelist.list_face[3]['name']))
                               face_img4.image(facelist.list_face[3]['pic'])

                            if len(facelist.list_face)>=5:
                               face_text5.text(str(facelist.list_face[4]['id'])+" "+str(facelist.list_face[4]['name']))
                               face_img5.image(facelist.list_face[4]['pic'])

                            if len(facelist.list_face)>=6:
                               face_text6.text(str(facelist.list_face[5]['id'])+" "+str(facelist.list_face[5]['name']))
                               face_img6.image(facelist.list_face[5]['pic'])

                            if len(facelist.list_face)>=7:
                               face_text7.text(str(facelist.list_face[6]['id'])+" "+str(facelist.list_face[6]['name']))
                               face_img7.image(facelist.list_face[6]['pic'])

                            if len(facelist.list_face)>=8:
                               face_text8.text(str(facelist.list_face[7]['id'])+" "+str(facelist.list_face[7]['name']))
                               face_img8.image(facelist.list_face[7]['pic'])

                            if len(facelist.list_face)>=9:
                               face_text9.text(str(facelist.list_face[8]['id'])+" "+str(facelist.list_face[8]['name']))
                               face_img9.image(facelist.list_face[8]['pic'])

                            if len(facelist.list_face)>=10:
                               face_text10.text(str(facelist.list_face[9]['id'])+" "+str(facelist.list_face[9]['name']))
                               face_img10.image(facelist.list_face[9]['pic'])

                            if len(facelist.list_face)>=11:
                               face_text11.text(str(facelist.list_face[10]['id'])+" "+str(facelist.list_face[10]['name']))
                               face_img11.image(facelist.list_face[10]['pic'])
                               
                            if len(facelist.list_face)>=12:
                               face_text12.text(str(facelist.list_face[11]['id'])+" "+str(facelist.list_face[11]['name']))
                               face_img12.image(facelist.list_face[11]['pic'])

                            if len(facelist.list_face)>=13:
                               face_text13.text(str(facelist.list_face[12]['id'])+" "+str(facelist.list_face[12]['name']))
                               face_img13.image(facelist.list_face[12]['pic'])

                            if len(facelist.list_face)>=14:
                               face_text14.text(str(facelist.list_face[13]['id'])+" "+str(facelist.list_face[13]['name']))
                               face_img14.image(facelist.list_face[13]['pic'])

                            if len(facelist.list_face)>=15:
                               face_text15.text(str(facelist.list_face[14]['id'])+" "+str(facelist.list_face[14]['name']))
                               face_img15.image(facelist.list_face[14]['pic'])

                            if len(facelist.list_face)>=16:
                               face_text16.text(str(facelist.list_face[15]['id'])+" "+str(facelist.list_face[15]['name']))
                               face_img16.image(facelist.list_face[15]['pic'])

                            if len(facelist.list_face)>=17:
                               face_text17.text(str(facelist.list_face[16]['id'])+" "+str(facelist.list_face[16]['name']))
                               face_img17.image(facelist.list_face[16]['pic'])

                            if len(facelist.list_face)>=18:
                               face_text18.text(str(facelist.list_face[17]['id'])+" "+str(facelist.list_face[17]['name']))
                               face_img18.image(facelist.list_face[17]['pic'])

                            if len(facelist.list_face)>=19:
                               face_text19.text(str(facelist.list_face[18]['id'])+" "+str(facelist.list_face[18]['name']))
                               face_img19.image(facelist.list_face[18]['pic'])

                            if len(facelist.list_face)>=20:
                               face_text20.text(str(facelist.list_face[19]['id'])+" "+str(facelist.list_face[19]['name']))
                               face_img20.image(facelist.list_face[19]['pic'])
        elif detec_sub_option == 'Video':
            input_path = st.text_input('Video Path')
            st.write('The current Video is', input_path)
            if input_path is not None and input_path !="":
                mask_model = torch.load("./torch_mask7.pt")
                mask_model.eval()
                resnet = InceptionResnetV1(pretrained='vggface2',classify=False,device='cuda').eval()
                Frame_video = st.image([])
                facelist = face_list()
                col1,col2,col3,col4,col5 = st.columns(5)
                with col1:
                    face_text1 = st.empty()
                    face_img1 = st.image([])
                    face_text6 = st.empty()
                    face_img6 = st.image([])
                    face_text11 = st.empty()
                    face_img11 = st.image([])
                    face_text16 = st.empty()
                    face_img16 = st.image([])
                with col2:
                    face_text2 = st.empty()
                    face_img2 = st.image([])
                    face_text7 = st.empty()
                    face_img7 = st.image([])
                    face_text12 = st.empty()
                    face_img12 = st.image([])
                    face_text17 = st.empty()
                    face_img17 = st.image([])
                with col3:
                    face_text3 = st.empty()
                    face_img3 = st.image([])
                    face_text8 = st.empty()
                    face_img8 = st.image([])
                    face_text13 = st.empty()
                    face_img13 = st.image([])
                    face_text18 = st.empty()
                    face_img18 = st.image([])
                with col4:
                    face_text4 = st.empty()
                    face_img4 = st.image([])
                    face_text9 = st.empty()
                    face_img9 = st.image([])
                    face_text14 = st.empty()
                    face_img14 = st.image([])
                    face_text19 = st.empty()
                    face_img19 = st.image([])
                with col5:
                    face_text5 = st.empty()
                    face_img5 = st.image([])
                    face_text10 = st.empty()
                    face_img10 = st.image([])
                    face_text15 = st.empty()
                    face_img15 = st.image([])
                    face_text20 = st.empty()
                    face_img20 = st.image([])
                imgtest = st.image([])

                cfg = get_config()
                cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
                attempt_download("deep_sort_pytorch/configs/deep_sort.yaml", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
                deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                    use_cuda=True)
                device = select_device('cuda')
                half = device.type != 'cpu'
                model = attempt_load("crowdhuman_yolov5m.pt", map_location=device)  
                stride = int(model.stride.max())  
                imgsz = check_img_size(640, s=stride)  
                names = model.module.names if hasattr(model, 'module') else model.names
                model.half()
                dataset = LoadImages(input_path,img_size=imgsz, stride=stride)
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
                t0 = time.time()
                for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    t1 = time_sync()
                    pred = model(img, augment= False)[0]
                    pred = non_max_suppression(
                        pred,0.8,0.5, classes=1, agnostic=False)
                    t2 = time_sync()
                    for i, det in enumerate(pred):
                        p, s, im0 = path, '', im0s
                        s += '%gx%g ' % img.shape[2:]
                        annotator = Annotator(im0, line_width=2, pil=not ascii)
                        if det is not None and len(det):
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                            xywhs = xyxy2xywh(det[:, 0:4])
                            confs = det[:, 4]
                            clss = det[:, 5]


                            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                            if len(outputs) > 0:
                                for j, (output, conf) in enumerate(zip(outputs, confs)): 
                                
                                    bboxes = output[0:4]
                                    id = output[4]
                                    cls = output[5]
                                    c = int(cls)  # integer class
                
                                                
                                    x,y,w,h=bboxes
                                    normal = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    # current_frame = annotator.result()
                                    current_frame = im0
                                    face = current_frame[y:h,x:w]
                                    face = cv.resize(face,(224,224))
                                    face = cv.cvtColor(face,cv.COLOR_BGR2RGB)
                                    face = np.expand_dims(face,axis=0)
                                    face = torch.tensor(face/255)
                                    face = face.permute(0,3,1,2)
                                    face = normal(face)
                                    face = face.float().to(device)
                                    result = mask_model(face)
                                    if result[0][0] > 0.70:
                                        # label = f'{id} {"masked"}'
                                        annotator.box_label(bboxes, color=(0,255,0))
                                    else:
                                        face2 = current_frame[y:h,x:w]
                                        face2 = cv.cvtColor(face2,cv.COLOR_BGR2RGB)
                                        face2 = cv.resize(face2,(160,160))
                                        face_save = face2
                                        face2 = np.expand_dims(face2,axis=0)
                                        face2 = torch.tensor(face2/255)
                                        face2 = face2.permute(0,3,1,2)
                                        face2 = face2.float().to(device)
                                        result2 = resnet(face2).cpu().detach().numpy()
                                        nameface = face_data_man(face = result2[0])
                                        # label = f'{id} {"Nomask"} {nameface}'
                                        facelist.insert(pic_face=face_save,id_face=id,name_face=nameface)
                                        annotator.box_label(bboxes, color=(0,0,255))
                        else:
                            deepsort.increment_ages()
                        im0 = annotator.result()
                        im0 = cv.cvtColor(im0,cv.COLOR_BGR2RGB)
                        Frame_video.image(im0)
                        if len(facelist.list_face)>0:
                            if len(facelist.list_face)>=1:
                               face_text1.text(str(facelist.list_face[0]['id'])+" "+str(facelist.list_face[0]['name']))
                               face_img1.image(facelist.list_face[0]['pic'])

                            if len(facelist.list_face)>=2:
                               face_text2.text(str(facelist.list_face[1]['id'])+" "+str(facelist.list_face[1]['name']))
                               face_img2.image(facelist.list_face[1]['pic'])

                            if len(facelist.list_face)>=3:
                               face_text3.text(str(facelist.list_face[2]['id'])+" "+str(facelist.list_face[2]['name']))
                               face_img3.image(facelist.list_face[2]['pic'])

                            if len(facelist.list_face)>=4:
                               face_text4.text(str(facelist.list_face[3]['id'])+" "+str(facelist.list_face[3]['name']))
                               face_img4.image(facelist.list_face[3]['pic'])

                            if len(facelist.list_face)>=5:
                               face_text5.text(str(facelist.list_face[4]['id'])+" "+str(facelist.list_face[4]['name']))
                               face_img5.image(facelist.list_face[4]['pic'])

                            if len(facelist.list_face)>=6:
                               face_text6.text(str(facelist.list_face[5]['id'])+" "+str(facelist.list_face[5]['name']))
                               face_img6.image(facelist.list_face[5]['pic'])

                            if len(facelist.list_face)>=7:
                               face_text7.text(str(facelist.list_face[6]['id'])+" "+str(facelist.list_face[6]['name']))
                               face_img7.image(facelist.list_face[6]['pic'])

                            if len(facelist.list_face)>=8:
                               face_text8.text(str(facelist.list_face[7]['id'])+" "+str(facelist.list_face[7]['name']))
                               face_img8.image(facelist.list_face[7]['pic'])

                            if len(facelist.list_face)>=9:
                               face_text9.text(str(facelist.list_face[8]['id'])+" "+str(facelist.list_face[8]['name']))
                               face_img9.image(facelist.list_face[8]['pic'])

                            if len(facelist.list_face)>=10:
                               face_text10.text(str(facelist.list_face[9]['id'])+" "+str(facelist.list_face[9]['name']))
                               face_img10.image(facelist.list_face[9]['pic'])

                            if len(facelist.list_face)>=11:
                               face_text11.text(str(facelist.list_face[10]['id'])+" "+str(facelist.list_face[10]['name']))
                               face_img11.image(facelist.list_face[10]['pic'])
                               
                            if len(facelist.list_face)>=12:
                               face_text12.text(str(facelist.list_face[11]['id'])+" "+str(facelist.list_face[11]['name']))
                               face_img12.image(facelist.list_face[11]['pic'])

                            if len(facelist.list_face)>=13:
                               face_text13.text(str(facelist.list_face[12]['id'])+" "+str(facelist.list_face[12]['name']))
                               face_img13.image(facelist.list_face[12]['pic'])

                            if len(facelist.list_face)>=14:
                               face_text14.text(str(facelist.list_face[13]['id'])+" "+str(facelist.list_face[13]['name']))
                               face_img14.image(facelist.list_face[13]['pic'])

                            if len(facelist.list_face)>=15:
                               face_text15.text(str(facelist.list_face[14]['id'])+" "+str(facelist.list_face[14]['name']))
                               face_img15.image(facelist.list_face[14]['pic'])

                            if len(facelist.list_face)>=16:
                               face_text16.text(str(facelist.list_face[15]['id'])+" "+str(facelist.list_face[15]['name']))
                               face_img16.image(facelist.list_face[15]['pic'])

                            if len(facelist.list_face)>=17:
                               face_text17.text(str(facelist.list_face[16]['id'])+" "+str(facelist.list_face[16]['name']))
                               face_img17.image(facelist.list_face[16]['pic'])

                            if len(facelist.list_face)>=18:
                               face_text18.text(str(facelist.list_face[17]['id'])+" "+str(facelist.list_face[17]['name']))
                               face_img18.image(facelist.list_face[17]['pic'])

                            if len(facelist.list_face)>=19:
                               face_text19.text(str(facelist.list_face[18]['id'])+" "+str(facelist.list_face[18]['name']))
                               face_img19.image(facelist.list_face[18]['pic'])

                            if len(facelist.list_face)>=20:
                               face_text20.text(str(facelist.list_face[19]['id'])+" "+str(facelist.list_face[19]['name']))
                               face_img20.image(facelist.list_face[19]['pic'])

    elif main_option == 'Upload Photo':
        st.write("Upload Photo:")
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        if image_file is not None:
            data_image = Image.open(image_file)
            st.image(data_image)
        st.write("Enter Name:")
        data_name =  st.text_input("")
        if image_file is not None:
            result = []
            img = None
            result,img = face_detect(image=data_image)
            faces = []
            count = 1
            if result is not None:
                for i in result:
                    faces.append(str(count))
                    count+=1
            else:
                st.write("Can't detected face !!")
            options = st.multiselect(
                    'Select your face :',
                    faces,
                    faces)
            if img is not None:
                if len(options) != 0:
                    for i in options:
                        x,y,w,h = result[int(i)-1]
                        calx = (w-x)*(10/100)
                        caly = (h-y)*(10/100)
                        x = x-calx
                        y = y-caly
                        w = w+calx
                        h = h+caly
                        cv.putText(img, f'{i}',(int(x),int(y)-5),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                        cv.rectangle(img,(int(x),int(y)),(int(w),int(h)),(255,0,0),2)
                st.image(img)
                if st.button("Upload"):
                    
                    if(len(options)!=1 and data_name == ""):
                        st.write("Pls select one face and Enter Name!!!!")
                    elif(len(options)!=1):
                        st.write("Pls select one face !!!!")
                    elif(data_name == ""):
                        st.write("Pls Enter Name !!!!")
                    else:
                        c_x,c_y,c_w,c_h = result[int(options[0])-1]
                        crop = img[int(c_y):int(c_h),int(c_x):int(c_w)]
                        st.write(face_data_man.upload_label(face=crop,name=data_name))


        
    

if __name__ == '__main__':
		main()	