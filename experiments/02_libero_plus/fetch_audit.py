"""Retrieve the CPU preparation audit without copying large cloud assets."""
from pathlib import Path
import modal

root = Path(__file__).parent / 'audit'
root.mkdir(exist_ok=True)
volume = modal.Volume.from_name('knightec-vla-stage1')
for name in ['assets_manifest.json', 'selection_audit.json', 'plus_tasks.json']:
    (root / name).write_bytes(b''.join(volume.read_file('plus/' + name)))
print(root)
