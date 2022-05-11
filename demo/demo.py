import time
import bisect
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec


VIDEO_FILE = 'bunny.mp4' # 'la_luna.mp4'
data_dir = "./data/"
session = "PL01-138cfd959876bdf2.log" # "PL01-c7180378e51b58de.log" # "PL01-5c6457b4e2c68d39.log" # "PL01-138cfd959876bdf2.log" # "PL01-4207998440a92c1c.log" # "PL01-4207998440a92c1c.log", "PL01-7f8a5a0ab378d455.log", "PL01-9c16d837e409887d.log", "PL01-7cb0405aa4b25f4d.log"
RL_BITRATE_FILE = data_dir + session + "_rl_bitrate"
RL_BUFFER_FILE = data_dir + session + '_rl_buffer'
MPC_BITRATE_FILE = data_dir + session + '_bb_bitrate'
MPC_BUFFER_FILE = data_dir + session + '_bb_buffer'
TRACE_FILE = data_dir + session + '_rl_network_trace'
demo_file = session + '_demo.mp4'
# RL_BITRATE_FILE = './rl_bitrate'
# RL_BUFFER_FILE = './rl_buffer'
# MPC_BITRATE_FILE = './bb_bitrate'
# MPC_BUFFER_FILE = './bb_buffer'
# TRACE_FILE = './network_trace'
SKIP_FRAMES = 200 # 8820
TOTAL_FRAMES = 1200 # 2000, better: 15 * seconds (x-axis in plot)
CHUNK_LEN = 4.0
# ALL_BITRATES = {0.3, 0.75, 1.2}
# ALL_BITRATES = {300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0}
ALL_BITRATES = {750.0, 2500.0, 4500.0, 6000.0, 8000.0, 20000.0}

cap = cv.VideoCapture(VIDEO_FILE)
kern_map = {0.3: np.ones((12, 12), np.float32) / 144, 
			0.75: np.ones((6, 6), np.float32) / 36, 
			1.2: np.ones((1, 1), np.float32) / 1}
kern_map = {300.0: np.ones((15, 15), np.float32) / 225, 
			750.0: np.ones((12, 12), np.float32) / 144, 
			1200.0: np.ones((9, 9), np.float32) / 81, 
			1850.0: np.ones((6, 6), np.float32) / 36, 
			2850.0: np.ones((3, 3), np.float32) / 9, 
			4300.0: np.ones((1, 1), np.float32) / 1}
kern_map = {750.0: np.ones((15, 15), np.float32) / 225, 
			2500.0: np.ones((12, 12), np.float32) / 144, 
			4500.0: np.ones((9, 9), np.float32) / 81, 
			6000.0: np.ones((6, 6), np.float32) / 36, 
			8000.0: np.ones((3, 3), np.float32) / 9, 
			20000.0: np.ones((1, 1), np.float32) / 1}

text_map = {0.3: '240P',
			0.75: '360P',
			1.2: '720P'}
text_map = {300.0: '240P',
			750.0: '360P',
			1200.0: '480P',
			1850.0: '720P',
			2850.0: '1080P',
			4300.0: '1440P'}
text_map = {750.0: '240P',
			2500.0: '360P',
			4500.0: '480P',
			6000.0: '720P',
			8000.0: '1080P',
			20000.0: '1440P'}

def read_file(FILE_NAME):
	ts = []
	vs = []
	with open(FILE_NAME, 'rb') as f:
		for line in f:
			parse = line.split()
			if len(parse) != 2:
				break
			ts.append(float(parse[0]))
			vs.append(float(parse[1]))
	return ts, vs

rl_bitrates_ts, rl_bitrates = read_file(RL_BITRATE_FILE)
rl_buffer_ts, rl_buffers = read_file(RL_BUFFER_FILE)
mpc_bitrates_ts, mpc_bitrates = read_file(MPC_BITRATE_FILE)
mpc_buffer_ts, mpc_buffers = read_file(MPC_BUFFER_FILE)
trace_ts, trace_bw = read_file(TRACE_FILE)

print " -- Processing videos -- "
all_processed_frames = {}
for br in ALL_BITRATES:
	all_processed_frames[br] = []

for _ in xrange(SKIP_FRAMES):
	_, frame = cap.read()	

# while(cap.isOpened()):
for f in xrange(TOTAL_FRAMES):
	print 'frame', f
	_, frame = cap.read()
	frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	for br in ALL_BITRATES:
		processed_frame = cv.filter2D(frame, -1, kern_map[br])
		all_processed_frames[br].append(processed_frame)

# frame = all_processed_frames[1.2][0]
frame = all_processed_frames[2500.0][0]


fig = plt.figure(figsize=(14, 6))

gs = gridspec.GridSpec(6, 6)
ax1 = plt.subplot(gs[:3, :3])
ax2 = plt.subplot(gs[3:, :3])
ax3 = plt.subplot(gs[:2, 3:])
ax4 = plt.subplot(gs[2:4, 3:])
ax5 = plt.subplot(gs[4:, 3:])

bbox_props = dict(boxstyle="round", fc="gray", ec="0.5", alpha=0.5)

img1 = ax1.imshow(frame)
ax1.set_ylabel('RL', size=21, color='#f23030')
ax1.xaxis.set_tick_params(bottom='off', labelbottom='off') 
ax1.yaxis.set_tick_params(left='off', labelleft='off')
text1 = ax1.text(1760, 1000, "240P", color="white", ha="center", va="center", size=16, bbox=bbox_props)

img2 = ax2.imshow(frame)
ax2.set_ylabel('BB', size=21, color='#2127dd')
ax2.xaxis.set_tick_params(bottom='off', labelbottom='off') 
ax2.yaxis.set_tick_params(left='off', labelleft='off')
text2 = ax2.text(1760, 1000, "240P", color="white", ha="center", va="center", size=16, bbox=bbox_props)

ax3.plot(rl_buffer_ts, rl_buffers, color='#f23030')
bar1, = ax3.plot([0, 0], [0, 45], color="orange", alpha=0.5)
ax3.set_ylabel('RL buffer (sec)')
ax3.set_xlim([-5, 120])
ax3.set_ylim([-2, 46])
ax3.xaxis.set_tick_params(labelbottom='off') 
# ax3.yaxis.set_tick_params(labelleft='off') 

ax4.plot(trace_ts, trace_bw, color='#1c1c1c', alpha=0.9)
bar2, = ax4.plot([0, 0], [0.4, 2.3], color="orange", alpha=0.5)
ax4.set_ylabel('Throughput (mbps)')
ax4.set_xlim([-5, 120])
ax4.xaxis.set_tick_params(labelbottom='off') 
# ax4.yaxis.set_tick_params(labelleft='off') 

ax5.plot(mpc_buffer_ts, mpc_buffers, color='#2127dd')
bar3, = ax5.plot([0, 0], [0, 45], color="orange", alpha=0.5)
ax5.set_ylabel('BB buffer (sec)')
ax5.set_xlim([-5, 120])
ax3.set_ylim([-2, 46])
ax5.xaxis.set_tick_params(labelbottom='off') 
# ax5.yaxis.set_tick_params(labelleft='off') 
ax5.set_xlabel('Time')

rolling_ts = np.linspace(0, 4 * len(rl_bitrates) - 4, len(rl_bitrates) * 20)
# 4 * 48 = 64 * 3 
# 4 * 30 =  40 * 3
rolling_ts = np.linspace(0, 4 * 30, 30 * 20)
# import pdb; pdb.set_trace()
def get_frame_quality(rolling_ts, bitrates_ts, bitrates, buffer_ts, buffers):
	frame_quality = {}
	text_quality = {}

	last_frame = 0
	for t in rolling_ts:
		br_pt = bisect.bisect(bitrates_ts, t) - 1
		buf_pt = bisect.bisect(buffer_ts, t) - 1

		if buffers[buf_pt] > 0.05: # 0.05:
			last_frame = (last_frame + 2) % TOTAL_FRAMES
		frame_quality[t] = all_processed_frames[bitrates[br_pt]][last_frame]
		text_quality[t] = text_map[bitrates[br_pt]]

	return frame_quality, text_quality

rl_frame_quality, rl_text_quality = get_frame_quality(
	rolling_ts, rl_bitrates_ts, rl_bitrates, rl_buffer_ts, rl_buffers)
mpc_frame_quality, mpc_text_quality = get_frame_quality(
	rolling_ts, mpc_bitrates_ts, mpc_bitrates, mpc_buffer_ts, mpc_buffers)

def animate(i):
	bar1.set_data([i, i], [0, 45])
	bar2.set_data([i, i], [0.4, 2.3])
	bar3.set_data([i, i], [0, 45])

	img1.set_data(rl_frame_quality[i])
	img2.set_data(mpc_frame_quality[i])

	text1.set_text(rl_text_quality[i])
	text2.set_text(mpc_text_quality[i])
	
	return bar1, bar2, bar3, img1, img2, text1, text2

ani = animation.FuncAnimation(fig, animate, rolling_ts, 
							  interval=50, blit=True)

# plt.show()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani.save(demo_file, writer=writer)
# with open('demo.mp4', 'wb') as f:
#     ani.save(f, writer=writer)
