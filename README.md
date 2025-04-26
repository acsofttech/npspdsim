# 🧠 NPS Robotics | PD Line-Follower Simulator Game

จำลองการทำงานของหุ่นยนต์ตามเส้นด้วยเซ็นเซอร์ 8 ตัว และระบบ PD Controller พร้อมสนามให้ทดลองเล่นจริง!

## ✅ Features
- PD Controller ปรับค่า Kp/Kd แบบเรียลไทม์
- ปรับสเกลการแสดงภาพสนามได้
- 8 Reflective Sensors
- ปรับระยะห่างระหว่างเซ็นเซอร์ได้
- ระบบ Checkpoint และ Finish Line
- ตรวจจับการหลุดเส้น
- แสดงเวลาวิ่งครบรอบ
- เลือกรูปแบบของสนามได้  ขนาดสนามที่แนะนำ 800x800 px
- มี 2 Checkpoint 1.ใช้แถบสี #FF00FF 2.ใช้แถบสี #FF8000 และเส้นชัยใช้แถบสี #0000FF

## 🚀 วิธีใช้งาน
bash
git clone https://github.com/yourname/pdsim.git
cd pdsim
pip install -r requirements.txt
streamlit run app.py
