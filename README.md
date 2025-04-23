# npspdsim
# 🧠 NPS PD Line-Follower Simulator

จำลองการทำงานของหุ่นยนต์ตามเส้นด้วยเซ็นเซอร์ 8 ตัว และระบบ PD Controller พร้อมสนามให้ทดลองเล่นจริง!

## ✅ Features
- PD Controller ปรับค่า Kp/Kd แบบเรียลไทม์
- 8 Reflective Sensors
- ระบบ Checkpoint และ Finish Line
- ตรวจจับการหลุดเส้น
- แสดงเวลาวิ่งครบรอบ
- รองรับภาพสนามจากโฟลเดอร์ `assets/`

## 🚀 วิธีใช้งาน
```bash
git clone https://github.com/yourname/pdsim.git
cd pdsim
pip install -r requirements.txt
streamlit run app.py
