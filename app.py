"""
NPS ROBOTICS PD Line-Follower Simulator
---------------------------------------
By. Amnart Plailaharn 
"""
import asyncio, math, time, base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
# ────────────────────────── Page / Layout ──────────────────────────
st.set_page_config(page_title="NPSROBOTICS PD Line-Follower", page_icon="🤖", layout="centered")

"""🤖 NPS ROBOTICS PD Line-Follower Game"""
# ────────────────────── Sidebar – Display & PD gains ─────────────────────
st.sidebar.header("🤖 NPS ROBOTICS – PDSIMBOT")
scale_pct = st.sidebar.slider("Display scale (%)", 40, 200, 100, 5)
SCALE     = scale_pct / 100
TRK_W, TRK_H = int(400 * SCALE), int(400 * SCALE)

# Robot shape & sensors
ROBOT_W = max(12, int(28 * SCALE))
ROBOT_H = max(8,  int(18 * SCALE))
S_R     = max(2,  int(4 * SCALE))

spacing_px = st.sidebar.slider("Sensor spacing (px)", 0, 30, 5)
SPACING    = int(spacing_px * SCALE)

Kp = st.sidebar.slider("Kp - Propertional", 0.0, 1.0, 0.000, 0.001, '%.3f')
Kd = st.sidebar.slider("Kd - Derivative", 0.0, 0.2, 0.020, 0.001, '%.3f')
base_speed = st.sidebar.number_input("Base speed (px/s)", 20, 255, 100,10)

# ─────────────────────────── Track asset ───────────────────────────
BASE_DIR = Path(__file__).parent
ASSETS   = BASE_DIR / "assets"; ASSETS.mkdir(exist_ok=True, parents=True)
DEFAULT_TRACK = ASSETS / "track_default.png"
if not list(ASSETS.glob("*.png")):
    img = Image.new("RGB", (800,800), "white")
    d   = ImageDraw.Draw(img)
    d.line((50, 300, 750, 300), fill="black", width=20)
    img.save(DEFAULT_TRACK)

track_choice = st.sidebar.selectbox("Track image",sorted(p.name for p in ASSETS.glob("*.png")))
TRACK_PATH = ASSETS / track_choice

track_img  = Image.open(TRACK_PATH).resize((TRK_W, TRK_H), Image.NEAREST)
TRACK_RGB  = np.asarray(track_img)
TRACK_GRAY = np.asarray(track_img.convert("L"))
buf = BytesIO(); track_img.save(buf, format="PNG")
TRACK_B64 = base64.b64encode(buf.getvalue()).decode()

# ───── Color masks (Finish & Checkpoints) ─────
FIN_MASK = (TRACK_RGB[:,:,2] > 200) & (TRACK_RGB[:,:,1] < 80) & (TRACK_RGB[:,:,0] < 80)
CP1_MASK = (TRACK_RGB[:,:,0] > 100) & (TRACK_RGB[:,:,2] > 200) & (TRACK_RGB[:,:,1] < 80)
CP2_MASK = (TRACK_RGB[:,:,0] > 200) & (TRACK_RGB[:,:,1] > 100) & (TRACK_RGB[:,:,2] < 80)
CP_MASKS = [CP1_MASK, CP2_MASK]
CP_TOTAL = len(CP_MASKS)

# ───────────────────────── Session-state ─────────────────────────
START_POS = {"x": 20*SCALE, "y": 200*SCALE, "theta": 0.0}
state = st.session_state
state.setdefault("running", False)
state.setdefault("robot",   START_POS.copy())
state.setdefault("check_idx", 0)
state.setdefault("lap_times", [])
state.setdefault("t_start", None)

OFFSETS = [-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]
WEIGHTS = [-4,  -3,  -2,  -1,  1,  2,  3,  4]
# ───── Checkpoint progress placeholders ─────
cp_bar  = st.progress(0.0)
cp_text = st.empty()

def reset_cp_ui():
    cp_bar.progress(0.0)
    cp_text.write(f"Checkpoint: 0/{CP_TOTAL}")

reset_cp_ui()
# ────────────────── Control buttons ──────────────────
colA, colB = st.columns(2)
with colA:
    if st.button("▶️ Start" if not state.running else "⏸ Pause",
                 use_container_width=True):
        if not state.running:                    
            state.t_start   = time.perf_counter()
            state.check_idx = 0
            reset_cp_ui()
        state.running = not state.running
with colB:
    if st.button("🔄 Reset", use_container_width=True):
        state.robot = START_POS.copy()
        state.running = False
        state.check_idx = 0
        reset_cp_ui()

# ───────────────────── Helper functions ─────────────────────
canvas_spot = st.empty()




def draw_overlay(x: float, y: float, theta: float):
    dx, dy = math.sin(theta), -math.cos(theta)
    circles = [f"<circle cx='{x+dx*off*SPACING:.1f}' "
               f"cy='{y+dy*off*SPACING:.1f}' r='{S_R}' fill='green'/>"
               for off in OFFSETS]
    svg = f"""
    <div style='position:relative;width:{TRK_W}px;height:{TRK_H}px;'>
      <img src='data:image/png;base64,{TRACK_B64}'
           style='position:absolute;width:{TRK_W}px;height:{TRK_H}px;' />
      <svg width='{TRK_W}' height='{TRK_H}'
           style='position:absolute;left:0;top:0;'>
        <rect x='{x-ROBOT_W/2:.1f}' y='{y-ROBOT_H/2:.1f}'
              width='{ROBOT_W}' height='{ROBOT_H}' fill='red'
              transform='rotate({math.degrees(theta):.1f},{x:.1f},{y:.1f})'/>
        {''.join(circles)}
      </svg>
    </div>"""
    canvas_spot.markdown(svg, unsafe_allow_html=True)

def sample_error(x,y,theta):
    dx,dy = math.sin(theta), -math.cos(theta)
    es = [w for w,off in zip(WEIGHTS,OFFSETS)
          if 0<=(sy:=int(y+dy*off*SPACING))<TRACK_GRAY.shape[0]
          and 0<=(sx:=int(x+dx*off*SPACING))<TRACK_GRAY.shape[1]
          and TRACK_GRAY[sy,sx]<=128]
    return sum(es)/len(es) if es else 0.0

def bicycle_update(x,y,th,v_l,v_r,dt,wheel_base=60*SCALE):
    v=(v_l+v_r)/2; omg=(v_r-v_l)/wheel_base
    return x+v*math.cos(th)*dt, y+v*math.sin(th)*dt, th+omg*dt

# ───────────────────── Simulation loop ─────────────────────
H,W = TRACK_GRAY.shape
def clamp(y,x): return int(max(0,min(H-1,y))), int(max(0,min(W-1,x)))

async def sim_loop():
    prev_err, t_prev = 0.0, time.perf_counter()
    oob_since = None
    while state.running:
        now=time.perf_counter(); dt=now-t_prev; t_prev=now
        r=state.robot
        err = sample_error(r["x"],r["y"],r["theta"])
        deriv=(err-prev_err)/dt if dt else 0
        u=Kp*err+Kd*deriv
        v_l,v_r = base_speed+u*base_speed, base_speed-u*base_speed
        r["x"],r["y"],r["theta"]=bicycle_update(r["x"],r["y"],r["theta"],v_l,v_r,dt)
        prev_err=err

        # OOB guard
        in_frame=(0<=r["x"]<W and 0<=r["y"]<H)
        if in_frame: oob_since=None
        else:
            oob_since=oob_since or now
            if now-oob_since>1.0:
                state.running=False
                st.warning("⛔ หลุดสนาม – ต้องรีเซ็ตใหม่นะจ๊ะ 😀")
                reset_cp_ui()
                break

        yy,xx = clamp(r["y"],r["x"])
        if state.check_idx<CP_TOTAL and CP_MASKS[state.check_idx][yy,xx]:
            state.check_idx+=1
            frac=state.check_idx/CP_TOTAL
            cp_bar.progress(frac)
            cp_text.write(f"Checkpoint: {state.check_idx}/{CP_TOTAL}")

        finished=(state.check_idx==CP_TOTAL) and FIN_MASK[yy,xx]
        if finished:
            lap=round(now-state.t_start,2)
            state.lap_times.append(lap)
            st.success(f"🏁 Lap time: {lap:.2f} s 👍")
            state.running=False
            reset_cp_ui()

        draw_overlay(r["x"],r["y"],r["theta"])
        await asyncio.sleep(0.016)

draw_overlay(**state.robot)

if state.running:
    asyncio.run(sim_loop())
