# 🧠 NPS Robotics | **PD Line-Follower Simulator Game**

จำลองการทำงานของหุ่นยนต์เดินตามเส้นแบบสมจริง!  
พร้อมระบบควบคุม PD Controller และเซ็นเซอร์ 8 ตัวในโลกเสมือนจริง 🎯

---

## ✅ Features
- ปรับค่า **Basespeed / Kp / Kd** แบบเรียลไทม์
- ปรับ **ขนาดการแสดงผล** ของสนามได้
- เซ็นเซอร์สะท้อนแสง **8 ตัว** (Reflective Sensors)
- ปรับ **ระยะห่างเซ็นเซอร์** ได้ตามต้องการ
- มีระบบ **Checkpoint** และ **เส้นชัย Finish Line**
- ตรวจจับ **การหลุดเส้น** อัตโนมัติ
- แสดง **เวลาวิ่งครบรอบ** (Lap Timing)
- รองรับการเลือก **รูปแบบสนาม** ขนาดแนะนำ **800×800 px**
- ระบบ Checkpoint ใช้สีมาตรฐาน:
  - Checkpoint 1 ➔ สี **#FF00FF** (ม่วงชมพู)
  - Checkpoint 2 ➔ สี **#FF8000** (ส้ม)
  - เส้นชัย ➔ สี **#0000FF** (น้ำเงิน)

---

## 🚀 วิธีติดตั้งและใช้งาน
```bash
git clone https://github.com/yourname/pdsim.git
cd pdsim
pip install -r requirements.txt
streamlit run app.py

