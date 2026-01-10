import cv2
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
alpha = 0.85        # EMA smoothing factor (0.7~0.9)
use_percentile = 50 # 50 = median
max_gain = 1.5      # 防止極端亮度爆掉
min_gain = 0.5

input_video = "log/eval/output.mp4"
output_video = "output_deflicker.mp4"

# -----------------------------
# Read video
# -----------------------------
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
luma_stats = []

print("Reading video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)

    stat = np.percentile(L, use_percentile)
    frames.append(lab)
    luma_stats.append(stat)

cap.release()

luma_stats = np.array(luma_stats)

# -----------------------------
# Temporal EMA smoothing
# -----------------------------
luma_smooth = np.zeros_like(luma_stats)
luma_smooth[0] = luma_stats[0]

for i in range(1, len(luma_stats)):
    luma_smooth[i] = (
        alpha * luma_stats[i]
        + (1 - alpha) * luma_smooth[i - 1]
    )

# -----------------------------
# Reference luminance
# -----------------------------
luma_ref = np.median(luma_smooth)

print(f"Reference luminance: {luma_ref:.2f}")

# -----------------------------
# Write output video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Processing frames...")

for lab, luma in zip(frames, luma_smooth):
    gain = luma_ref / (luma + 1e-6)
    gain = np.clip(gain, min_gain, max_gain)

    L = lab[:, :, 0].astype(np.float32)
    L = L * gain
    L = np.clip(L, 0, 255)

    lab[:, :, 0] = L.astype(np.uint8)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    writer.write(bgr)

writer.release()

print("Done! Output saved to:", output_video)
