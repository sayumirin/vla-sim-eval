"""Pin the published LIBERO-PRO paraphrases without installing its simulator."""
import hashlib
import json
from pathlib import Path
import subprocess
import yaml

root = Path(__file__).parent
repo = root / 'LIBERO-PRO'
source = repo / 'libero_ood/ood_language.yaml'
prompts = yaml.safe_load(source.read_text(encoding='utf-8'))['libero_goal']
assert len(prompts) == 10 and all(len(v) == 3 for v in prompts.values())
result = {'source_repo': 'https://github.com/Zxy-MLlab/LIBERO-PRO',
          'source_revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
          'source_file': 'libero_ood/ood_language.yaml',
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'selection_rule': 'First published paraphrase per original task, fixed before evaluation.',
          'paraphrases': {k: v[0] for k, v in prompts.items()}, 'all_candidates': prompts}
(root / 'configs').mkdir(exist_ok=True)
(root / 'configs/paraphrases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
