"""Budget-capped ACT data-scaling study on Modal."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).parent
COMMIT = '72e1d9e5e76ee719bacfc78bc34d817bee5fb521'
app = modal.App('knightec-vla-stage3b')
volume = modal.Volume.from_name('knightec-vla-stage3b', create_if_missing=True)
base = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git','build-essential','ffmpeg','libegl1','libgl1',
                 'libgles2','libgl1-mesa-dri','libglib2.0-0','libosmesa6')
    .pip_install('cmake>=3.29,<4','ninja')
    .run_commands(
        'ln -sf /usr/local/lib/python3.12/site-packages/cmake/data/bin/cmake /usr/local/bin/cmake',
        'git init /opt/lerobot',
        'git -C /opt/lerobot remote add origin https://github.com/huggingface/lerobot.git',
        f'git -C /opt/lerobot fetch --depth 1 origin {COMMIT}',
        f'git -C /opt/lerobot checkout --detach {COMMIT}')
    .pip_install('/opt/lerobot[libero,evaluation,pi]','torch==2.11.0','torchvision==0.26.0','cmake<4')
    .apt_install('libopengl0')
    .pip_install('gym-aloha==0.1.5')
    .env({'HF_HOME':'/tmp/rq1-hf','TORCH_HOME':'/opt/rq1/torch_cache','MUJOCO_GL':'egl',
          'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_TELEMETRY':'1',
          'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
    .add_local_file(ROOT/'configs/rq1_plan.json','/opt/rq1/plan.json',copy=True)
    .add_local_file(ROOT/'build_data.py','/opt/rq1/build_data.py',copy=True)
)


def build_cache():
    import subprocess
    import sys
    subprocess.run([sys.executable,'/opt/rq1/build_data.py'],check=True,timeout=1100)


data_image_recipe = base.run_function(build_cache,cpu=2,memory=8192,timeout=1200)
# Immutable CPU-built cache verified in logs/prepare.log. To rebuild from scratch,
# replace the next line with: data_image = data_image_recipe
data_image = modal.Image.from_id('im-P1njbgQP5D1ZOK2fJcUo8Z').pip_install('/opt/lerobot[training]','accelerate==1.14.0')
image = (data_image
    .add_local_file(ROOT/'train_act.py','/opt/rq1/train_act.py',copy=True)
    .add_local_file(ROOT/'evaluate_act.py','/opt/rq1/evaluate_act.py',copy=True))


@app.function(image=image,volumes={'/data':volume},cpu=1,memory=2048,timeout=120,scaledown_window=2)
def inspect_data():
    import importlib.metadata
    import inspect
    import gym_aloha.env
    result = json.loads(Path('/opt/rq1/data_audit.json').read_text())
    result['gym_aloha_version'] = importlib.metadata.version('gym-aloha')
    result['env_source'] = inspect.getsource(gym_aloha.env.AlohaEnv)
    Path('/data/data_audit.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.function(image=image,volumes={'/data':volume},cpu=(2,2),memory=(8192,8192),timeout=300,scaledown_window=2)
def cpu_check():
    import datetime
    import os
    import subprocess
    import sys
    directory=Path('/data/cpu_checks')/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    plan=json.loads(Path('/opt/rq1/plan.json').read_text())
    command=[sys.executable,'/opt/rq1/train_act.py','--policy.type=act','--policy.device=cpu',
        f'--dataset.repo_id={plan["dataset_repo"]}','--dataset.root=/opt/rq1/dataset',
        f'--dataset.revision={plan["dataset_revision"]}',f'--dataset.episodes={json.dumps(plan["subsets"]["5"])}',
        '--dataset.video_backend=pyav','--policy.push_to_hub=false','--wandb.enable=false',
        '--batch_size=1','--num_workers=0','--steps=1','--seed=777','--env_eval_freq=0',
        '--save_checkpoint=false','--log_freq=1',f'--output_dir={directory/"train"}']
    try:
        with (directory/'train.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180,
                env={**os.environ,'RQ1_OUTPUT':str(directory),'RQ1_START_STEP':'0'})
        payload={'output':str(directory),'exit_code':result.returncode,
                 'log_tail':(directory/'train.log').read_text()[-6500:]}
        if result.returncode==0:
            code="import gymnasium as gym, gym_aloha, json; e=gym.make('gym_aloha/AlohaTransferCube-v0',obs_type='pixels_agent_pos'); o,i=e.reset(seed=20000); v=e.step(o['agent_pos']); print(json.dumps({'image_shape':list(o['pixels']['top'].shape),'state_shape':list(o['agent_pos'].shape),'reward':float(v[1]),'info':str(v[-1])})); e.close()"
            simulation=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60,
                env={**os.environ,'MUJOCO_GL':'osmesa','PYOPENGL_PLATFORM':'osmesa'})
            payload['simulation_check']={'exit_code':simulation.returncode,'stdout':simulation.stdout,'stderr':simulation.stderr[-2000:]}
    finally:
        volume.commit()
    return payload


def run_job(size,seed,steps,kind,gpu,limit,eval_every=0,resume_from=''):
    import datetime
    import hashlib
    import os
    import signal
    import subprocess
    import sys
    import time
    started = time.monotonic()
    volume.reload()
    plan = json.loads(Path('/opt/rq1/plan.json').read_text())
    assert str(size) in plan['subsets'] and steps > 0 and kind in {'smoke','pilot','formal'}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path('/data/results')/f'{kind}_{gpu}_n{size}_s{seed}_k{steps}_{stamp}'
    output.mkdir(parents=True)
    run = {'kind':kind,'gpu_requested':gpu,'n_demonstrations':size,'train_seed':seed,
           'steps':steps,'selected_episodes':plan['subsets'][str(size)],'output':str(output),
           'status':'running','dataset_revision':plan['dataset_revision'],'lerobot_commit':COMMIT,
           'config':plan,'limit_seconds':limit,'resume_from':resume_from,
           'source_hashes':{name:hashlib.sha256((Path('/opt/rq1')/name).read_bytes()).hexdigest()
                            for name in ['train_act.py','evaluate_act.py','plan.json']}}
    (output/'manifest.json').write_text(json.dumps(run,indent=2))
    (output/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    run['gpu_actual'] = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    train_dir = output/'train'
    command = [sys.executable,'/opt/rq1/train_act.py', '--policy.type=act',
        f'--dataset.repo_id={plan["dataset_repo"]}','--dataset.root=/opt/rq1/dataset',
        f'--dataset.revision={plan["dataset_revision"]}',
        f'--dataset.episodes={json.dumps(plan["subsets"][str(size)])}',
        '--dataset.video_backend=pyav','--env.type=aloha','--env.task=AlohaTransferCube-v0',
        '--policy.device=cuda','--policy.push_to_hub=false','--wandb.enable=false',
        '--batch_size=8','--num_workers=4',f'--steps={steps}',f'--seed={seed}',
        f'--env_eval_freq={eval_every}',f'--save_freq={eval_every if eval_every else 0}',
        '--log_freq=100','--eval.batch_size=10','--eval.n_episodes=20',
        f'--output_dir={train_dir}']
    start_step = 0
    if resume_from:
        previous = json.loads((Path(resume_from)/'manifest.json').read_text())
        assert previous['status']=='completed' and previous['kind']=='pilot'
        assert previous['n_demonstrations']==size and previous['train_seed']==seed
        start_step = previous['steps']
        config_path = Path(resume_from)/'train/checkpoints/last/pretrained_model/train_config.json'
        assert config_path.is_file()
        command = [sys.executable,'/opt/rq1/train_act.py',f'--config_path={config_path}',
            '--resume=true',f'--steps={steps}',f'--output_dir={train_dir}',
            f'--env_eval_freq={eval_every}',f'--save_freq={eval_every}']
    run['train_command'] = command
    run['start_step'] = start_step

    def execute(cmd,log_name,extra_env):
        remaining = limit-(time.monotonic()-started)-30
        if remaining<=0:
            raise TimeoutError('Run budget exhausted')
        with (output/log_name).open('w') as log:
            child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                     env={**os.environ,**extra_env})
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                raise TimeoutError(log_name+' exceeded reserved runtime')
        if code:
            raise RuntimeError(log_name+f' exited {code}: '+(output/log_name).read_text()[-5000:])

    try:
        execute(command,'train.log',{'RQ1_OUTPUT':str(output),'RQ1_START_STEP':str(start_step)})
        checkpoint = train_dir/'checkpoints/last/pretrained_model'
        assert (checkpoint/'model.safetensors').is_file()
        run['checkpoint'] = str(checkpoint)
        run['checkpoint_relative'] = str(checkpoint.resolve().relative_to(output.resolve()))
        run['checkpoint_sha256'] = hashlib.file_digest(open(checkpoint/'model.safetensors','rb'),'sha256').hexdigest()
        run['train_seconds'] = time.monotonic()-started
        if kind != 'pilot':
            n = 2 if kind=='smoke' else 50
            eval_seed = 20000 if kind=='smoke' else 10000
            eval_output = output/'eval'
            metadata = {'n_demonstrations':size,'train_seed':seed,'training_steps':steps,
                'eval_seed_start':eval_seed,'dataset_revision':plan['dataset_revision'],
                'lerobot_commit':COMMIT,'model':'act','checkpoint_sha256':run['checkpoint_sha256']}
            eval_command = [sys.executable,'/opt/rq1/evaluate_act.py',f'--policy.path={checkpoint}',
                '--policy.device=cuda','--env.type=aloha','--env.task=AlohaTransferCube-v0',
                f'--eval.batch_size={min(10,n)}',f'--eval.n_episodes={n}',f'--seed={eval_seed}',
                f'--output_dir={eval_output}']
            run['eval_command'] = eval_command
            execute(eval_command,'eval.log',{'RQ1_EVAL_OUTPUT':str(eval_output),
                'RQ1_EVAL_METADATA':json.dumps(metadata)})
            run['results'] = json.loads((eval_output/'eval_info.json').read_text())
        else:
            run['pilot_evaluations'] = {p.stem:json.loads(p.read_text()) for p in output.glob('pilot_eval_step*.json')}
        run['status']='completed'
    except Exception as exc:
        run['status']='timed_out' if isinstance(exc,TimeoutError) else 'failed'
        run['error']=str(exc)
    finally:
        run['measured_seconds']=time.monotonic()-started
        (output/'manifest.json').write_text(json.dumps(run,indent=2))
        volume.commit()
        run['wall_seconds']=time.monotonic()-started
    return run


def run_eval_check(gpu,source):
    import datetime
    import os
    import subprocess
    import sys
    import time
    started=time.monotonic()
    volume.reload()
    old=json.loads((Path(source)/'manifest.json').read_text())
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=Path('/data/results')/f'evalcheck_{gpu}_{stamp}'
    output.mkdir(parents=True)
    run={'kind':'eval_check','gpu_requested':gpu,'output':str(output),'source':source,
         'status':'running','n_demonstrations':50,'train_seed':777,'steps':200}
    (output/'manifest.json').write_text(json.dumps(run,indent=2))
    metadata={'n_demonstrations':50,'train_seed':777,'training_steps':200,
              'eval_seed_start':20000,'model':'act','checkpoint_sha256':old['checkpoint_sha256']}
    command=[sys.executable,'/opt/rq1/evaluate_act.py',f'--policy.path={old["checkpoint"]}',
        '--policy.device=cuda','--env.type=aloha','--env.task=AlohaTransferCube-v0',
        '--eval.batch_size=10','--eval.n_episodes=20','--seed=20000',f'--output_dir={output/"eval"}']
    try:
        with (output/'eval.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=500,
                env={**os.environ,'RQ1_EVAL_OUTPUT':str(output/'eval'),'RQ1_EVAL_METADATA':json.dumps(metadata)})
        run['status']='completed' if result.returncode==0 else 'failed'
        if (output/'eval/eval_info.json').exists():
            run['results']=json.loads((output/'eval/eval_info.json').read_text())
        if result.returncode:
            run['error']=(output/'eval.log').read_text()[-5000:]
    except Exception as exc:
        run.update(status='failed',error=str(exc))
    finally:
        run['wall_seconds']=time.monotonic()-started
        (output/'manifest.json').write_text(json.dumps(run,indent=2))
        volume.commit()
    return run


COMMON = dict(image=image,volumes={'/data':volume},cpu=(4,4),memory=(16384,16384),
              retries=0,scaledown_window=2,startup_timeout=120,single_use_containers=True,max_containers=4)
L4_COMMON={**COMMON,'cpu':(2,2)}
FORMAL_COMMON={**COMMON,'max_containers':5}

@app.function(**COMMON,gpu='A10',timeout=600)
def eval_check_a10(source:str):return run_eval_check('A10',source)

@app.function(**COMMON,gpu='A100-40GB',timeout=600)
def eval_check_a100(source:str):return run_eval_check('A100',source)

@app.function(**COMMON,gpu='A10',timeout=900)
def smoke_a10(): return run_job(50,777,200,'smoke','A10',840)

@app.function(**COMMON,gpu='A100-40GB',timeout=900)
def smoke_a100(): return run_job(50,777,200,'smoke','A100',840)

@app.function(**L4_COMMON,gpu='L4',timeout=900)
def smoke_l4(): return run_job(50,777,200,'smoke','L4',840)

@app.function(**COMMON,gpu='A10',timeout=3600)
def pilot_a10(steps:int=6000,resume_from:str=''):
    return run_job(50,777,steps,'pilot','A10',3540,2000,resume_from)

@app.function(**COMMON,gpu='A100-40GB',timeout=3600)
def pilot_a100(steps:int=6000,resume_from:str=''):
    return run_job(50,777,steps,'pilot','A100',3540,2000,resume_from)

@app.function(**L4_COMMON,gpu='L4',timeout=3600)
def pilot_l4(steps:int=6000,resume_from:str=''):
    return run_job(50,777,steps,'pilot','L4',3540,2000,resume_from)

@app.function(**COMMON,gpu='A10',timeout=1800)
def formal_a10_short(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','A10',1740)

@app.function(**FORMAL_COMMON,gpu='A10',timeout=1200)
def formal_a10_compact(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','A10',1140)

@app.function(**COMMON,gpu='A10',timeout=3600)
def formal_a10_long(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','A10',3540)

@app.function(**COMMON,gpu='A100-40GB',timeout=1800)
def formal_a100_short(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','A100',1740)

@app.function(**COMMON,gpu='A100-40GB',timeout=3600)
def formal_a100_long(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','A100',3540)

@app.function(**L4_COMMON,gpu='L4',timeout=1800)
def formal_l4_short(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','L4',1740)

@app.function(**L4_COMMON,gpu='L4',timeout=2400)
def formal_l4_medium(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','L4',2340)

@app.function(**L4_COMMON,gpu='L4',timeout=3600)
def formal_l4_long(size:int,seed:int,steps:int): return run_job(size,seed,steps,'formal','L4',3540)


def reserve_many(requests):
    ledger_path=ROOT/'budget_ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit_usd':13.5,'user_credit_usd':15,'jobs':[]}
    committed=sum(j.get('settled_estimate_usd',j['reserved_usd']) for j in ledger['jobs'])
    entries=[]
    for label,gpu,timeout in requests:
        rate={'L4':.000222,'A10':.000306,'A100':.000583}[gpu]+(2 if gpu=='L4' else 4)*.0000131+16*.00000222
        entries.append({'label':label,'gpu':gpu,'timeout_seconds':timeout,
                        'usd_per_second':rate,'reserved_usd':(timeout+122)*rate,'state':'reserved'})
    total=committed+sum(e['reserved_usd'] for e in entries)
    assert total<=ledger['limit_usd'], f'Budget reservation rejected: {total:.2f} > {ledger["limit_usd"]}'
    tokens=list(range(len(ledger['jobs']),len(ledger['jobs'])+len(entries)))
    ledger['jobs'].extend(entries)
    ledger_path.write_text(json.dumps(ledger,indent=2))
    return tokens


def reserve(label,gpu,timeout):
    return reserve_many([(label,gpu,timeout)])[0]


def save_result(result,token):
    dest=ROOT/'modal_results'
    dest.mkdir(exist_ok=True)
    (dest/f'{Path(result["output"]).name}.json').write_text(json.dumps(result,indent=2))
    path=ROOT/'budget_ledger.json'
    ledger=json.loads(path.read_text())
    entry=ledger['jobs'][token]
    entry['settled_estimate_usd']=min(entry['reserved_usd'],(result['wall_seconds']+122)*entry['usd_per_second'])
    entry.update(state=result['status'],run=Path(result['output']).name)
    path.write_text(json.dumps(ledger,indent=2))
    print(json.dumps({k:result.get(k) for k in ['output','status','train_seconds','wall_seconds','error']},indent=2),flush=True)
    return result['status']=='completed'


@app.local_entrypoint()
def main(mode:str='prepare',gpu:str='A10',steps:int=6000,resume_from:str=''):
    import concurrent.futures
    assert gpu in {'L4','A10','A100'}
    if mode=='cpu-check':
        result=cpu_check.remote()
        (ROOT/'cpu_check.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        assert result['exit_code']==0 and result.get('simulation_check',{}).get('exit_code')==0
        return
    if mode=='eval-check':
        requests=[('A10',eval_check_a10,'smoke_A10_n50_s777_k200_20261030T220520Z'),
                  ('A100',eval_check_a100,'smoke_A100_n50_s777_k200_20261030T220520Z')]
        tokens=reserve_many([('eval_check',g,600) for g,_,_ in requests])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(fn.remote,'/data/results/'+name):token for (_,fn,name),token in zip(requests,tokens)}
            results=[save_result(f.result(),futures[f]) for f in concurrent.futures.as_completed(futures)]
        assert all(results),'Evaluation check failed'
        return
    if mode=='prepare':
        data=inspect_data.remote()
        (ROOT/'data_audit.json').write_text(json.dumps(data,indent=2))
        print(json.dumps({k:v for k,v in data.items() if k!='env_source'},indent=2),flush=True)
        return
    if mode=='smoke':
        tasks=[('L4',smoke_l4),('A10',smoke_a10),('A100',smoke_a100)]
        reservations=reserve_many([('smoke',g,900) for g,_ in tasks])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(fn.remote):token for (_,fn),token in zip(tasks,reservations)}
            ok=[]
            for future in concurrent.futures.as_completed(futures):
                ok.append(save_result(future.result(),futures[future]))
        assert all(ok),'Inspect smoke results before continuing.'
    elif mode=='pilot':
        token=reserve('pilot',gpu,3600)
        function={'L4':pilot_l4,'A10':pilot_a10,'A100':pilot_a100}[gpu]
        assert save_result(function.remote(steps,resume_from),token)
    elif mode=='formal':
        design=json.loads((ROOT/'configs/final_design.json').read_text())
        gpu=design['gpu']
        duration=design['timeout_seconds']
        function={('A10',1200):formal_a10_compact,('A10',1800):formal_a10_short,('A10',3600):formal_a10_long,
                  ('A100',1800):formal_a100_short,('A100',3600):formal_a100_long,
                  ('L4',1800):formal_l4_short,('L4',2400):formal_l4_medium,('L4',3600):formal_l4_long}[(gpu,duration)]
        jobs=[(n,s) for s in design['training_seeds'] for n in design['sizes']]
        tokens=reserve_many([(f'formal_n{n}_s{s}',gpu,duration) for n,s in jobs])
        assert 1<=design.get('max_parallel',4)<=5
        with concurrent.futures.ThreadPoolExecutor(max_workers=design.get('max_parallel',4)) as pool:
            futures={pool.submit(function.remote,n,s,design['steps']):t for (n,s),t in zip(jobs,tokens)}
            ok=[]
            for future in concurrent.futures.as_completed(futures):
                ok.append(save_result(future.result(),futures[future]))
        assert all(ok),'Some formal runs did not finish; do not silently replace outcomes.'
    else:
        raise ValueError('Use prepare, smoke, pilot or formal')
