"""CL Benchmark: Continual Learning baselines vs ContinualWAM on LIBERO."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DISPLAY", "")
import argparse, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, od, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(od,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,ad))
    def forward(self, x): return self.net(x)

class RSSM(nn.Module):
    def __init__(self, od, ad, h=128, z=16):
        super().__init__()
        self.obs_enc=nn.Linear(od,h); self.act_enc=nn.Linear(ad,h)
        self.prior=nn.Linear(h,z*2); self.obs_dec=nn.Linear(h+z,od)
        self.rnn=nn.GRUCell(h,h); self.h=h
    def train_loss(self, obs, act):
        B,T,D=obs.shape; h=torch.zeros(B,self.h,device=obs.device); s=0.0
        for t in range(T):
            oe=F.relu(self.obs_enc(obs[:,t])); ae=F.relu(self.act_enc(act[:,t]))
            h=self.rnn(oe+ae,h); ph=self.prior(h); pm,ps=ph.chunk(2,-1)
            z=pm+ps.exp()*torch.randn_like(pm); r=self.obs_dec(torch.cat([h,z],-1))
            s+=F.mse_loss(r,obs[:,t])
        return s/T
    def predict_error(self, o, a, no):
        with torch.no_grad():
            oe=F.relu(self.obs_enc(o)); ae=F.relu(self.act_enc(a)); h=oe+ae
            ph=self.prior(h); pm,_=ph.chunk(2,-1); r=self.obs_dec(torch.cat([h,pm],-1))
            return F.mse_loss(r,no,reduction="none").mean(-1)

class EMATrust:
    def __init__(self, a=0.1): self.a=a; self.avg=None
    def score(self, e):
        if self.avg is None: self.avg=e.mean().item()
        else: self.avg=(1-self.a)*self.avg+self.a*e.mean().item()
        return float(np.exp(-self.avg))

def flatten_obs(d):
    return np.concatenate([np.asarray(v,dtype=np.float32).flatten() for k,v in d.items() if "image" not in k])

def collect(suite, ti, nep=5, ms=50):
    from libero.libero.envs import OffScreenRenderEnv
    env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(ti),camera_heights=64,camera_widths=64)
    od=len(flatten_obs(env.reset())); ad=env.env.action_dim
    oa,aa=[],[]
    for _ in range(nep):
        o=flatten_obs(env.reset())
        for _ in range(ms):
            a=np.random.randn(ad).astype(np.float32); r=env.step(a); oa.append(o); aa.append(a); o=flatten_obs(r[0])
    env.close()
    return torch.tensor(np.array(oa),dtype=torch.float32),torch.tensor(np.array(aa),dtype=torch.float32),od,ad

def eval_acc(model, tasks, ti):
    return [(-F.mse_loss(model(tasks[tj][0]),tasks[tj][1]).item()) for tj in range(ti+1)]

def run_seqft(tasks, od, ad, dev, ne=20):
    m=MLP(od,ad).to(dev); o=torch.optim.Adam(m.parameters(),lr=3e-4); r=[]
    for ti,(ob,ab) in enumerate(tasks):
        for _ in range(ne): l=F.mse_loss(m(ob),ab); o.zero_grad(); l.backward(); o.step()
        r.append(eval_acc(m,tasks,ti))
    return r

def run_ewc(tasks, od, ad, dev, ne=20, lam=100):
    m=MLP(od,ad).to(dev); o=torch.optim.Adam(m.parameters(),lr=3e-4); fish={}; prev={}; r=[]
    for ti,(ob,ab) in enumerate(tasks):
        if ti>0: fish[ti-1]={n:p.grad.data.clone() for n,p in m.named_parameters() if p.grad is not None}; prev[ti-1]={n:p.data.clone() for n,p in m.named_parameters()}
        for _ in range(ne):
            l=F.mse_loss(m(ob),ab)
            if ti>0 and ti-1 in fish:
                e=sum((fish[ti-1][n]*(p-prev[ti-1][n]).pow(2)).sum() for n,p in m.named_parameters()); l=l+lam*e
            o.zero_grad(); l.backward(); o.step()
        r.append(eval_acc(m,tasks,ti))
    return r

def run_lwf(tasks, od, ad, dev, ne=20, lam=1.0):
    m=MLP(od,ad).to(dev); o=torch.optim.Adam(m.parameters(),lr=3e-4); pm=None; r=[]
    for ti,(ob,ab) in enumerate(tasks):
        if ti>0: pm=copy.deepcopy(m)
        for _ in range(ne):
            p=m(ob); l=F.mse_loss(p,ab)
            if pm is not None:
                with torch.no_grad(): op=pm(ob)
                l=l+lam*F.mse_loss(p,op)
            o.zero_grad(); l.backward(); o.step()
        r.append(eval_acc(m,tasks,ti))
    return r

def run_er(tasks, od, ad, dev, ne=20, br=0.2):
    m=MLP(od,ad).to(dev); o=torch.optim.Adam(m.parameters(),lr=3e-4); bo,ba=[],[]; r=[]
    for ti,(ob,ab) in enumerate(tasks):
        bo.append(ob); ba.append(ab)
        for _ in range(ne):
            ao=torch.cat(bo); aa=torch.cat(ba); n=len(ao); p=torch.randperm(n)[:int(n*br)+len(ob)]
            l=F.mse_loss(m(ao[p]),aa[p]); o.zero_grad(); l.backward(); o.step()
        r.append(eval_acc(m,tasks,ti))
    return r

def run_packnet(tasks, od, ad, dev, ne=20):
    m=MLP(od,ad).to(dev); fm={}; r=[]
    for ti,(ob,ab) in enumerate(tasks):
        o=torch.optim.Adam(m.parameters(),lr=3e-4)
        for ep in range(ne):
            l=F.mse_loss(m(ob),ab); o.zero_grad(); l.backward(); o.step()
            if ep==ne-1 and ti<len(tasks)-1:
                with torch.no_grad():
                    for n,p in m.named_parameters():
                        if p.grad is not None: th=torch.quantile(p.grad.abs().flatten(),0.2); fm[n]=(p.grad.abs()>th).float()
        with torch.no_grad():
            for n,p in m.named_parameters():
                if n in fm: p.data*=(1-fm[n])
        r.append(eval_acc(m,tasks,ti))
    return r

def run_cwam(tasks, od, ad, dev, trust="ema", ne=20, lam=100):
    bb=RSSM(od,ad).to(dev); m=MLP(od,ad).to(dev)
    o=torch.optim.Adam(m.parameters(),lr=3e-4); bo=torch.optim.Adam(bb.parameters(),lr=3e-4)
    tr=EMATrust() if trust=="ema" else None; fish={}; prev={}; r=[]
    for ti,(ob,ab) in enumerate(tasks):
        T=min(32,len(ob)); ns=len(ob)//T
        if ns>0:
            obs=ob[:ns*T].view(ns,T,-1); acts=ab[:ns*T].view(ns,T,-1)
            for _ in range(10): l=bb.train_loss(obs,acts); bo.zero_grad(); l.backward(); bo.step()
        if ti>0: fish[ti-1]={n:p.grad.data.clone() for n,p in m.named_parameters() if p.grad is not None}; prev[ti-1]={n:p.data.clone() for n,p in m.named_parameters()}
        for _ in range(ne):
            p=m(ob); l=F.mse_loss(p,ab)
            if ti>0 and ti-1 in fish:
                with torch.no_grad():
                    e=bb.predict_error(ob[:-1],ab[:-1],ob[1:]); tw=tr.score(e) if tr else 1.0
                ewc=sum((fish[ti-1][n]*(p-prev[ti-1][n]).pow(2)).sum() for n,p in m.named_parameters()); l=l+lam*tw*ewc
            o.zero_grad(); l.backward(); o.step()
        r.append(eval_acc(m,tasks,ti))
    return r

def metrics(res):
    T=len(res); R=np.zeros((T,T))
    for i in range(T):
        for j in range(len(res[i])): R[i,j]=res[i][j]
        for j in range(len(res[i]),T): R[i,j]=R[i,j-1] if j>0 else 0
    auc=np.mean([R[i,i] for i in range(T)])
    fwt=np.mean([R[i,i] for i in range(1,T)]) if T>1 else 0
    nbt=np.mean([R[T-1,j]-R[j,j] for j in range(T-1)]) if T>1 else 0
    return {"auc":float(auc),"fwt":float(fwt),"nbt":float(nbt),"final":float(R[T-1,T-1])}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--suite",default="libero_spatial")
    p.add_argument("--n-episodes",type=int,default=5)
    p.add_argument("--max-steps",type=int,default=50)
    p.add_argument("--n-epochs",type=int,default=20)
    p.add_argument("--output",default="cl_benchmark_results.json")
    a=p.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from libero.libero import benchmark
    bd=benchmark.get_benchmark_dict(); s=bd[a.suite](); nt=s.n_tasks
    print(f"Benchmark: {a.suite} ({nt} tasks), Device: {dev}")
    td=[]; od=ad=0
    for ti in range(nt):
        ob,ab,o,d=collect(s,ti,a.n_episodes,a.max_steps); od=o; ad=d
        td.append((ob.to(dev),ab.to(dev)))
        print(f"  Task {ti}: {s.get_task(ti).name[:50]}... ({len(ob)} samples)")
    print(f"Obs dim: {od}, Act dim: {ad}")
    run = wandb.init(project="continualwam", name=f"cl-{a.suite}",
        tags=["cl-benchmark", a.suite],
        config={"suite": a.suite, "n_epochs": a.n_epochs}, reinit=True)
    ms={
        "SeqFT":   lambda: run_seqft(td,od,ad,dev,a.n_epochs),
        "ER":      lambda: run_er(td,od,ad,dev,a.n_epochs),
        "EWC":     lambda: run_ewc(td,od,ad,dev,a.n_epochs),
        "LwF":     lambda: run_lwf(td,od,ad,dev,a.n_epochs),
        "PackNet": lambda: run_packnet(td,od,ad,dev,a.n_epochs),
        "CWAM-EMA":   lambda: run_cwam(td,od,ad,dev,"ema",a.n_epochs),
        "CWAM-Multi": lambda: run_cwam(td,od,ad,dev,"multi_step",a.n_epochs),
    }
    ar={}
    for nm,fn in ms.items():
        t0=time.time(); res=fn(); el=time.time()-t0; m=metrics(res)
        ar[nm]={"metrics":m,"per_task":res,"time":el}
        print(f"  {nm}: AUC={m['auc']:.4f} FWT={m['fwt']:.4f} NBT={m['nbt']:.4f} Final={m['final']:.4f} ({el:.1f}s)")
        run.log({f"{nm}/auc": m["auc"], f"{nm}/fwt": m["fwt"], f"{nm}/nbt": m["nbt"], f"{nm}/final": m["final"]})
    with open(a.output,"w") as f: json.dump(ar,f,indent=2)
    names = list(ms.keys())
    log_bar_chart(run, "auc_comparison", names, [ar[n]["metrics"]["auc"] for n in names], "AUC by Method")
    log_bar_chart(run, "nbt_comparison", names, [ar[n]["metrics"]["nbt"] for n in names], "Forgetting (NBT)")
    log_bar_chart(run, "final_accuracy", names, [ar[n]["metrics"]["final"] for n in names], "Final Accuracy")
    log_bar_chart(run, "training_time", names, [ar[n]["time"] for n in names], "Training Time (s)")
    run.finish()
    print(f"\nSaved to {a.output}")
    print(f"\n{'Method':<16} {'AUC':>8} {'FWT':>8} {'NBT':>8} {'Final':>8}")
    print("-"*52)
    for nm in ms:
        m=ar[nm]["metrics"]
        print(f"{nm:<16} {m['auc']:>8.4f} {m['fwt']:>8.4f} {m['nbt']:>8.4f} {m['final']:>8.4f}")

if __name__=="__main__": main()
