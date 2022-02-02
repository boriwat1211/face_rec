โปรเจคดังกล่าวเป็นวิชา machine learning ปีที่ 3 เทอมที่ 1 ปีการศึกษา 2564 มหาวิทยาลัยพระจอมเกล้าพระนครเหนือ

ข้อมูลโปรเจค
เป็นโปรเจคการจับใบหน้าเเละตรวจสอบการใส่เเมสเเละจดจำใบหน้าว่าคนที่ไม่ใส่เเมสเป็นใครในฐานข้อมูล

วิธีการใช้งาน 
1) Clone โปรเจค YoloV5+Deepsort จาก https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
2) Copy Web.py ไว้ในโฟลเดอร์ของ VoloV5+Deepsort 
3) โหลดโมเดลจับการใส่เเมสจาก https://drive.google.com/file/d/1r_8HFfqVImC0e8cIDFKw9IhzF71VWNyI/view?usp=sharing
4) โหลดโมเดลจับใบหน้าจาก https://drive.google.com/file/d/1M5OXvDYI20f3g0p5i3nnGdQRnr7iIKN9/view?usp=sharing
5) นำโมเดลทั้งสองไว้ในโฟลเดอร์เดียวกับ Web.py
6) เริ่มการใช้งานโดยการใช้คำสั่ง streamlit run Web.py
