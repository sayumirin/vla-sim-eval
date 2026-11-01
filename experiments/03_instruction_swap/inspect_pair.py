"""Illustrate the first paired case using equally indexed saved video frames."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent
config = json.loads((root / 'configs/rq2_swap.json').read_text())
control = root / 'modal_results/C0_original_100_seed0_20261030T190322Z/videos/libero_goal_0/eval_episode_0.mp4'
swapped = root / 'inspection/C2_task0_episode0.mp4'
frames = [0, 40, 80, 120]
fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), layout='constrained')
for row, (condition, path) in enumerate([('C0_original', control), ('C2_swap', swapped)]):
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > max(frames)
    for col, index in enumerate(frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok
        axes[row, col].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Saved frame {index}', fontsize=10)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        for spine in axes[row,col].spines.values():
            spine.set_visible(False)
    axes[row,0].set_ylabel('Original' if row == 0 else 'Swapped', fontsize=12)
    capture.release()
fig.suptitle('Same initial scene, different instruction\nOriginal: '+config['tasks'][0]['instructions']['C0_original']+'\nSwapped: '+config['tasks'][0]['instructions']['C2_swap'], fontsize=13)
fig.supxlabel('Task 0, episode 0 • equally indexed saved frames\nIllustrative example; only original-task success is scored, not replacement-task completion.', fontsize=9)
target = root / 'inspection/paired_instruction_example.png'
fig.savefig(target, dpi=160)
plt.close(fig)
(root / 'inspection/paired_case.json').write_text(json.dumps({
    'task_id': 0, 'episode': 0, 'frame_indices': frames,
    'selection': 'First task and first episode; illustrative, not a representative performance estimate.',
    'original_instruction': config['tasks'][0]['instructions']['C0_original'],
    'swapped_instruction': config['tasks'][0]['instructions']['C2_swap'],
    'source_videos': [str(control.relative_to(root)), str(swapped.relative_to(root))],
    'limitation': 'Replacement-task success is not evaluated; video frame times are not simulation wall-clock times.'}, indent=2))
print(target)
