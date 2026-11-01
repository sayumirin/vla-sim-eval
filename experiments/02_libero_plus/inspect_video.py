"""Create a diagnostic contact sheet of saved simulation video frames."""
import argparse
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('video', type=Path)
args = parser.parse_args()
capture = cv2.VideoCapture(str(args.video))
assert capture.isOpened()
count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
for ax, index in zip(axes, [0, count//3, 2*count//3, count-1], strict=True):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    assert ok
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Frame {index} / {index/fps:.1f}s')
    ax.axis('off')
fig.suptitle(args.video.stem)
dest = args.video.with_suffix('.png')
fig.savefig(dest, dpi=120)
plt.close(fig)
capture.release()
print(dest)
