"""Build the side-by-side README GIFs from raw rollout videos.

The videos are not in git (they live in the Modal volumes and in local run folders), so this
script takes their paths as arguments. Needs opencv-python, Pillow and matplotlib (for the font).

python scripts/make_gifs.py swap  LEFT.mp4 RIGHT.mp4
python scripts/make_gifs.py act   LEFT.mp4 RIGHT.mp4
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
BOLD = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans-Bold.ttf'
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
GOOD, BAD = (0, 99, 0), (176, 40, 40)

# Each clip: video path argument, header lines, footer text and colour once the episode ends.
CLIPS = {
    'swap': dict(output='swap_side_by_side.gif', size=(288, 288), stride=3,
                 title='Same initial scene, same policy, different instruction',
                 headers=['"open the middle drawer of the cabinet"', '"put the bowl on the stove"'],
                 subheaders=['original instruction', 'swapped in from another task'],
                 endings=[('original task done (step 122)', GOOD),
                          ('original task not done (300-step limit)', BAD)]),
    'act': dict(output='act_5_vs_35_demos.gif', size=(320, 240), stride=4,
                title='ACT on AlohaTransferCube-v0, same test case, training seed 0',
                headers=['trained on 5 demonstrations', 'trained on 35 demonstrations'],
                subheaders=['', ''],
                endings=[('cube not transferred (400-step limit)', BAD),
                         ('cube transferred (step 215)', GOOD)]),
}


def frames(path, size, stride):
    capture = cv2.VideoCapture(str(path))
    result = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result.append(Image.fromarray(rgb).resize(size, Image.LANCZOS))
        index += 1
    capture.release()
    assert result, f'No frames read from {path}'
    return result


def build(kind, left, right):
    spec = CLIPS[kind]
    width, height = spec['size']
    gap, margin, footer = 10, 10, 30
    header = 74 if any(spec['subheaders']) else 58
    clips = [frames(left, spec['size'], spec['stride']), frames(right, spec['size'], spec['stride'])]
    length = max(map(len, clips)) + 12  # hold the final frame briefly before looping
    title_font = ImageFont.truetype(str(BOLD), 14)
    font = ImageFont.truetype(str(FONT), 13)
    small = ImageFont.truetype(str(FONT), 11)
    canvas_size = (2 * width + gap + 2 * margin, header + height + footer)
    images = []
    for t in range(length):
        canvas = Image.new('RGB', canvas_size, SURFACE)
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 8), spec['title'], font=title_font, fill=INK)
        for side, clip in enumerate(clips):
            x = margin + side * (width + gap)
            draw.text((x, 32), spec['headers'][side], font=font, fill=INK)
            draw.text((x, 51), spec['subheaders'][side], font=small, fill=INK_2)
            canvas.paste(clip[min(t, len(clip) - 1)], (x, header))
            if t >= len(clip) - 1:
                text, colour = spec['endings'][side]
                draw.text((x, header + height + 8), text, font=font, fill=colour)
        images.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    output = ROOT / 'figures' / spec['output']
    images[0].save(output, save_all=True, append_images=images[1:], duration=80, loop=0, optimize=True)
    print(f'{output.name}: {len(images)} frames, {output.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=sorted(CLIPS))
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    build(args.kind, args.left, args.right)
