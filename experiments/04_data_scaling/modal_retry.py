"""One fixed-seed infrastructure replacement; separately reserve until parent exits."""
import json
from pathlib import Path
import modal
from modal_act import COMMON, ROOT, run_job
app=modal.App('knightec-vla-stage3b-retry')

@app.function(**{**COMMON,'max_containers':1},gpu='A10',timeout=1800)
def replace_slow_attempt():
    return run_job(20,2,6000,'formal','A10',1740)

@app.local_entrypoint()
def main():
    path=ROOT/'retry_budget_ledger.json'
    assert not path.exists(), 'Never automatically duplicate this replacement'
    ledger=json.loads((ROOT/'budget_ledger.json').read_text())
    exposure=sum(j.get('settled_estimate_usd',j['reserved_usd']) for j in ledger['jobs'])
    rate=.00039392
    entry={'label':'infrastructure_retry_n20_s2','gpu':'A10','timeout_seconds':1800,
        'usd_per_second':rate,'reserved_usd':(1800+122)*rate,'state':'reserved',
        'replaces':'formal_A10_n20_s2_k6000_20261030T232234Z',
        'reason':'Original instance projected to exceed deadline from measured compute speed, before any evaluation result'}
    assert exposure+entry['reserved_usd']+15*422*rate<=13.5
    path.write_text(json.dumps(entry,indent=2))
    result=replace_slow_attempt.remote()
    name=Path(result['output']).name
    (ROOT/'modal_results'/f'{name}.json').write_text(json.dumps(result,indent=2))
    entry.update(state=result['status'],run=name,
        settled_estimate_usd=min(entry['reserved_usd'],(result['wall_seconds']+122)*rate))
    path.write_text(json.dumps(entry,indent=2))
    print(json.dumps({k:result.get(k) for k in ['output','status','wall_seconds','error']},indent=2),flush=True)
    assert result['status']=='completed'
