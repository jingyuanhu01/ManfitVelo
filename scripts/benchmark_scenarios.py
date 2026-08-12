"""Shared scenario/fitting-variant library for the simulation/ benchmark suite.

Despite the module's older "run_*" naming history (renamed from
run_field_informed_manfit_benchmark.py during the 2026-08-12 repo cleanup),
this is a LIBRARY, not a one-off script: most of simulation/'s active
benchmark scripts import scenario generators (`vector_data`, `scalar_data`,
`hairpin`, `SETS`), the fitting-variant dispatcher (`fit_vmf_variant`), the
position-only baseline (`position_only_trajectory` -- migrated in from the
retired `run_position_only_manfit_diagnostic.py`), and evaluation helpers
(`distance_to_manifold`, `geometry`, `field_metrics`) from here. The file's
own `main()`/`checks()`/`build_report()` at the bottom are a legacy
standalone smoke-test entry point (`python -m scripts.benchmark_scenarios`),
predating and independent of the `simulation/` pipeline's own reporting
(`simulation/build_experiment_report.py`) -- not part of any active
protocol, kept for its own self-consistency checks.
"""
from __future__ import annotations
import argparse,base64,inspect,json,os,sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib");os.environ.setdefault("LOKY_MAX_CPU_COUNT","8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.pca_denoisers import global_pca_denoise,local_pca_denoise,project_vectors_with_pca_info
from scripts.scalar_potential_manfit import estimate_gradient_from_neighbors
from scripts.velocity_manifold_fitter import VelocityManifoldFitter

EPS=1e-12;KGRID=(20,30,50,80,120);THETA=(.02,.05,.10,.20,.40);ETA=(.20,.35,.50,.70);TGRID=(3,5,8);KAPPA=(0.,.25,.50,1.,2.)
HAIRPIN_LEGACY_SEPARATION=.13
HAIRPIN_DEFAULT_SEPARATION=.22
VECTOR=("circle","s_curve","curved_hairpin","flat_rotation_annulus","half_sphere_tangent","y_branch","near_intersection")
SCALAR=("scalar_s_curve","scalar_saddle")
VMETHODS=("noisy_input","relaxed_global_pca","local_pca","position_only_manfit","vmf")
SMETHODS=("noisy_input","relaxed_global_pca","local_pca","position_only_manfit")
LABEL={"noisy_input":"Noisy input","relaxed_global_pca":"Global PCA","local_pca":"Local PCA","position_only_manfit":"Position-only MANFIT","vmf":"VMF"}
SETUP={
 "circle":"A one-dimensional closed curve with tangential flow and normal position noise.",
 "s_curve":"A genuinely one-dimensional S-shaped curve with a varying tangent direction.",
 "curved_hairpin":"A folded curve with nearby curved arms, incompatible flow, and fixed branch-aware targets.",
 "flat_rotation_annulus":"A flat annulus in ambient three-space with rotational flow and normal position noise.",
 "half_sphere_tangent":"A half-sphere carrying a projected non-axis-aligned tangent field with low-speed points excluded.",
 "y_branch":"A labeled Y-branch with outward flow and fixed samplewise clean targets.",
 "near_intersection":"Two labeled, nonintersecting nearby curves carry incompatible directions of flow.",
 "scalar_s_curve":"A scalar function with mostly nonzero intrinsic gradient is observed on a one-dimensional S-curve.",
 "scalar_saddle":"A scalar potential is observed on a saddle surface and evaluated against its intrinsic ambient gradient.",
}

@dataclass(frozen=True)
class Set:
 n:int;px:float;field_noise:float;d:int
SETS={s:Set(360,.05,.10,1) for s in ("circle","s_curve")};SETS|={"curved_hairpin":Set(480,.025,.10,1),"flat_rotation_annulus":Set(420,.05,.10,2),"half_sphere_tangent":Set(480,.04,.10,2),"y_branch":Set(480,.02,.10,1),"near_intersection":Set(480,.02,.10,1),"scalar_s_curve":Set(360,.05,.08,1),"scalar_saddle":Set(500,.05,.08,2),"swiss_roll":Set(480,.05,.10,2),"saddle_surface":Set(480,.05,.10,2)}

def norm(V):return V/(np.linalg.norm(V,axis=1,keepdims=True)+EPS)
def add_noise(X,F,N,s,rng):return X+rng.normal(scale=s.px,size=(len(X),1))*N,F+rng.normal(scale=s.field_noise,size=F.shape)

def _y_branch_finish(X0,V0,labels,ambiguity,position_noise,velocity_noise,extra_dims,seed,meta):
 # Inlined from the retired scripts/ambiguity_simulations.py (that file had
 # shrunk to just this helper + y_branch_outward_flow after the other two
 # generators it once held were confirmed unused outside archive/).
 rng=np.random.default_rng(seed+991); n=len(X0)
 X=np.hstack([X0+rng.normal(scale=position_noise,size=X0.shape),rng.normal(scale=position_noise,size=(n,extra_dims))])
 V=np.hstack([V0+rng.normal(scale=velocity_noise,size=V0.shape),rng.normal(scale=velocity_noise,size=(n,extra_dims))])
 return {"X":X,"V":V,"X_gt":X0,"V_gt":V0,"structure_labels":np.asarray(labels),"ambiguity_region":np.asarray(ambiguity,bool),**meta}

def y_branch_outward_flow(n_samples=600,position_noise=.01,velocity_noise=.05,extra_dims=12,seed=0):
 rng=np.random.default_rng(seed); counts=[n_samples//3,n_samples//3,n_samples-2*(n_samples//3)]
 t0=np.sort(rng.uniform(0,1,counts[0])); stem=np.c_[np.zeros_like(t0),-t0]; Vs=np.tile([0.,1.],(len(t0),1))
 t1=np.sort(rng.uniform(0,1,counts[1])); left=np.c_[-.8*t1,.8*t1]; Vl=np.tile(np.array([-.8,.8])/np.sqrt(1.28),(len(t1),1))
 t2=np.sort(rng.uniform(0,1,counts[2])); right=np.c_[.8*t2,.8*t2]; Vr=np.tile(np.array([.8,.8])/np.sqrt(1.28),(len(t2),1))
 X=np.vstack([stem,left,right]); V=np.vstack([Vs,Vl,Vr]); labels=np.r_[np.zeros(len(stem),int),np.ones(len(left),int),np.full(len(right),2,int)]; latent=np.r_[t0,t1,t2]; ambiguity=latent<.3
 return _y_branch_finish(X,V,labels,ambiguity,position_noise,velocity_noise,extra_dims,seed,{"latent_branch_coordinate":latent,"difficulty_value":position_noise,"minimum_structure_separation":np.nan,"scenario":"y_branch_outward_flow"})

def hairpin(n,sep,seed):
 rng=np.random.default_rng(seed);L=1.2;r=sep/2;curv=.15;length=np.array([2*L,np.pi*r,2*L]);cnt=np.maximum(4,np.floor(n*length/length.sum()).astype(int));cnt[0]+=n-cnt.sum();x1=np.sort(rng.uniform(-L,L,cnt[0]));phase=lambda x:np.pi*(x+L)/(2*L);shape=lambda x:curv*(1-np.cos(phase(x)));der=lambda x:curv*np.sin(phase(x))*np.pi/(2*L)
 lo=np.c_[x1,-r+shape(x1),np.zeros(len(x1))];v1=norm(np.c_[np.ones(len(x1)),der(x1),np.zeros(len(x1))]);a=np.sort(rng.uniform(-np.pi/2,np.pi/2,cnt[1]));cy=shape(np.array([L]))[0];turn=np.c_[L+r*np.cos(a),cy+r*np.sin(a),np.zeros(len(a))];v_turn=np.c_[-np.sin(a),np.cos(a),np.zeros(len(a))];x3=np.sort(rng.uniform(-L,L,cnt[2]))[::-1];up=np.c_[x3,r+shape(x3),np.zeros(len(x3))];v3=-norm(np.c_[np.ones(len(x3)),der(x3),np.zeros(len(x3))]);X=np.vstack([lo,turn,up]);V=np.vstack([v1,v_turn,v3]);N=norm(np.c_[-V[:,1],V[:,0],np.zeros(n)]);arc=np.r_[x1+L,2*L+(a+np.pi/2)*r,2*L+np.pi*r+(L-x3)];labels=np.r_[np.zeros(len(lo),int),np.ones(len(turn),int),np.full(len(up),2,int)];return X,V,N,arc,labels

def vector_data(name,seed,override_n=None,position_noise=None,velocity_noise=None):
 s=SETS[name];s=Set(override_n or s.n,s.px if position_noise is None else float(position_noise),s.field_noise if velocity_noise is None else float(velocity_noise),s.d);rng=np.random.default_rng(seed);n=s.n
 if name=="circle":t=rng.uniform(0,2*np.pi,n);X=np.c_[np.cos(t),np.sin(t),np.zeros(n)];F=np.c_[-np.sin(t),np.cos(t),np.zeros(n)];N=X.copy();labels=np.zeros(n,int)
 elif name=="s_curve":t=rng.uniform(-1.4,1.4,n);X=np.c_[np.sin(1.6*t),t,np.zeros(n)];F=norm(np.c_[1.6*np.cos(1.6*t),np.ones(n),np.zeros(n)]);N=norm(np.c_[-F[:,1],F[:,0],np.zeros(n)]);labels=np.zeros(n,int)
 elif name=="curved_hairpin":X,F,N,_,labels=hairpin(n,HAIRPIN_DEFAULT_SEPARATION,seed)
 elif name=="flat_rotation_annulus":r=np.sqrt(rng.uniform(.35**2,1,n));t=rng.uniform(0,2*np.pi,n);X=np.c_[r*np.cos(t),r*np.sin(t),np.zeros(n)];F=np.c_[-X[:,1],X[:,0],np.zeros(n)];N=np.tile([0.,0.,1.],(n,1));labels=np.zeros(n,int)
 elif name=="half_sphere_tangent":
  vals=[]
  while len(vals)<n:
   z=rng.uniform(0,1);a=rng.uniform(0,2*np.pi);x=np.array([np.sqrt(1-z*z)*np.cos(a),np.sqrt(1-z*z)*np.sin(a),z]);b=np.array([.7,-.4,.6]);v=b-x*np.dot(b,x)
   if np.linalg.norm(v)>.18:vals.append((x,v))
  X=np.array([q[0] for q in vals]);raw=np.array([q[1] for q in vals]);F=raw/np.mean(np.linalg.norm(raw,axis=1));N=X.copy();labels=np.zeros(n,int)
 elif name=="y_branch":z=y_branch_outward_flow(n,0,0,1,seed);X=np.c_[z["X_gt"],np.zeros(n)];F=np.c_[z["V_gt"],np.zeros(n)];labels=z["structure_labels"];N=norm(np.c_[-F[:,1],F[:,0],np.zeros(n)])
 elif name=="near_intersection":n1=n//2;x1=rng.uniform(-1.1,1.1,n1);x2=rng.uniform(-1.1,1.1,n-n1);sep=.13;c=.3;X=np.vstack([np.c_[x1,sep/2+c*x1*x1,np.zeros(n1)],np.c_[x2,-sep/2-c*x2*x2,np.zeros(n-n1)]]);F=np.vstack([norm(np.c_[np.ones(n1),2*c*x1,np.zeros(n1)]),-norm(np.c_[np.ones(n-n1),-2*c*x2,np.zeros(n-n1)])]);labels=np.r_[np.zeros(n1,int),np.ones(n-n1,int)];N=norm(np.c_[-F[:,1],F[:,0],np.zeros(n)])
 elif name=="swiss_roll":
  # Group-A regular 2D manifold: a genuinely curved sheet, complementing the
  # flat annulus and the positive-curvature half-sphere. t is the spiral/roll
  # parameter, y is the flat "width" direction. Velocity flows outward along
  # increasing t (unrolling flow). One full winding (Δt=2π): more turns
  # (e.g. the classic 1.5-turn sklearn range) make Euclidean kNN bridge
  # across adjacent windings even at small k, so no method (including
  # ManfitVelo) beats noisy input anywhere in a reasonable k range --
  # verified empirically (simulation/log.md) that one winding is the
  # gentlest choice that still stays recognizably a rolled sheet while
  # giving every regular-scenario method a fair chance, matching this
  # scenario's role as a Group-A benchmark rather than an extra stress test.
  tmax=3.5*np.pi;t=rng.uniform(1.5*np.pi,tmax,n);y=rng.uniform(-1,1,n)
  X=np.c_[t*np.cos(t)/tmax,y,t*np.sin(t)/tmax]
  dXdt=np.c_[(np.cos(t)-t*np.sin(t))/tmax,np.zeros(n),(np.sin(t)+t*np.cos(t))/tmax]
  F=norm(dXdt);N=norm(np.c_[-dXdt[:,2],np.zeros(n),dXdt[:,0]]);labels=np.zeros(n,int)
 elif name=="saddle_surface":
  # Group-A regular 2D manifold with negative/mixed Gaussian curvature,
  # contrasting the positive-curvature half-sphere. a matches the curvature
  # scale already used by the legacy scalar_saddle scenario. Velocity flows
  # along +u (the "rising" ridge direction).
  a=.45;u=rng.uniform(-1,1,n);v=rng.uniform(-1,1,n)
  X=np.c_[u,v,a*(u*u-v*v)]
  Ju=np.c_[np.ones(n),np.zeros(n),2*a*u]
  F=norm(Ju);N=norm(np.c_[-2*a*u,2*a*v,np.ones(n)]);labels=np.zeros(n,int)
 Y,W=add_noise(X,F,N,s,rng);return {"Y":Y,"field":W,"P":X,"truth":F,"labels":labels,"d":s.d,"set":s,"true_normal":N}

def scalar_data(name,seed,override_n=None):
 s=SETS[name];s=Set(override_n or s.n,s.px,s.field_noise,s.d);rng=np.random.default_rng(seed);n=s.n
 if name=="scalar_s_curve":t=rng.uniform(-1.35,1.35,n);X=np.c_[np.sin(1.6*t),t,np.zeros(n)];tan=np.c_[1.6*np.cos(1.6*t),np.ones(n),np.zeros(n)];speed=np.linalg.norm(tan,axis=1);S=t+.18*np.sin(2*t);df=1+.36*np.cos(2*t);G=(df/speed)[:,None]*norm(tan);N=norm(np.c_[-tan[:,1],tan[:,0],np.zeros(n)]);labels=np.zeros(n,int)
 elif name=="scalar_saddle":u=rng.uniform(-1,1,n);v=rng.uniform(-1,1,n);a=.45;X=np.c_[u,v,a*(u*u-v*v)];Ju=np.c_[np.ones(n),np.zeros(n),2*a*u];Jv=np.c_[np.zeros(n),np.ones(n),-2*a*v];N=norm(np.cross(Ju,Jv));beta=.4;gamma=.35;S=.5*u*u+v*v+beta*u*v+gamma*u;du=u+beta*v+gamma;dv=2*v+beta*u;g11=(Ju*Ju).sum(1);g22=(Jv*Jv).sum(1);g12=(Ju*Jv).sum(1);det=g11*g22-g12*g12;coef1=(g22*du-g12*dv)/det;coef2=(-g12*du+g11*dv)/det;G=coef1[:,None]*Ju+coef2[:,None]*Jv;labels=np.zeros(n,int)
 Y,_=add_noise(X,np.zeros_like(X),N,s,rng);Sobs=S+rng.normal(scale=s.field_noise,size=n);return {"Y":Y,"scalar":Sobs,"P":X,"truth":G,"scalar_clean":S,"labels":labels,"d":s.d,"set":s}

@lru_cache(maxsize=None)
def manifold_support(name):
 """A seed-independent dense support used only for distance-to-manifold evaluation."""
 if name in ("circle",):
  t=np.linspace(0,2*np.pi,3000,endpoint=False);return np.c_[np.cos(t),np.sin(t),np.zeros(len(t))]
 if name in ("s_curve","scalar_s_curve"):
  t=np.linspace(-1.4,1.4,3500);return np.c_[np.sin(1.6*t),t,np.zeros(len(t))]
 if name=="curved_hairpin":
  X,_,_,_,_=hairpin(5000,HAIRPIN_DEFAULT_SEPARATION,918273);return X
 if name=="y_branch":
  t=np.linspace(0,1,1800);return np.vstack([np.c_[np.zeros_like(t),-t,np.zeros_like(t)],np.c_[-.8*t,.8*t,np.zeros_like(t)],np.c_[.8*t,.8*t,np.zeros_like(t)]])
 if name=="near_intersection":
  t=np.linspace(-1.1,1.1,3500);return np.vstack([np.c_[t,.13/2+.3*t*t,np.zeros_like(t)],np.c_[t,-.13/2-.3*t*t,np.zeros_like(t)]])
 if name=="scalar_saddle":
  u,v=np.meshgrid(np.linspace(-1,1,220),np.linspace(-1,1,220));return np.c_[u.ravel(),v.ravel(),.45*(u.ravel()**2-v.ravel()**2)]
 raise KeyError(name)

@lru_cache(maxsize=None)
def manifold_index(name):return NearestNeighbors(n_neighbors=1).fit(manifold_support(name))

def distance_to_manifold(name,X):
 if name=="flat_rotation_annulus":
  radial=np.sqrt(X[:,0]**2+X[:,1]**2);inplane=np.maximum.reduce([.35-radial,radial-1.,np.zeros(len(X))]);dist=np.sqrt(inplane**2+X[:,2]**2)
 elif name=="half_sphere_tangent":
  radius=np.linalg.norm(X,axis=1);upper=X[:,2]>=0;dist=np.empty(len(X));dist[upper]=abs(radius[upper]-1);rho=np.sqrt(X[~upper,0]**2+X[~upper,1]**2);dist[~upper]=np.sqrt((rho-1)**2+X[~upper,2]**2)
 else:dist=manifold_index(name).kneighbors(X)[0][:,0]
 return float(np.sqrt(np.mean(dist**2)))

def geometry(name,X,P):return distance_to_manifold(name,X),float(np.sqrt(np.mean(np.sum((X-P)**2,axis=1))))
def field_metrics(E,T):keep=np.linalg.norm(T,axis=1)>1e-10;nrmse=float(np.sqrt(np.sum((E-T)**2)/(np.sum(T**2)+EPS)));cos=np.sum(E[keep]*T[keep],axis=1)/((np.linalg.norm(E[keep],axis=1)+EPS)*np.linalg.norm(T[keep],axis=1));return nrmse,float(np.mean(np.clip(cos,-1,1))),float(keep.mean())

def full_global_pca_projection(Y,d):return global_pca_denoise(Y,d,return_info=True)
def relaxed_pca(Y,d,eta,T):proj,info=full_global_pca_projection(Y,d);factor=(1-eta)**T;return proj+factor*(Y-proj),info
def common_gradient(X,S,k,ridge):return estimate_gradient_from_neighbors(X,S,n_neighbors=min(k,len(X)-1),ridge=ridge)

def effective_sample_size(weights):
 weights=np.asarray(weights,float);return 1./np.maximum(np.sum(weights**2,axis=1),EPS)

def position_only_trajectory(X,V,d,k,T,eta_g,max_step_frac=.2,beta=1.0,eps=1e-12):
    """Spatial-only normal mean-shift trajectory; never imports or invokes VMF.

    This is the position-only restriction of the existing normal-update rule:
    Euclidean neighbors are frozen at t=0; local kernel weights/tangents are
    recomputed after every update; motion is restricted to the estimated normal.

    Migrated here 2026-08-12 (repo cleanup) from the retired
    `scripts/run_position_only_manfit_diagnostic.py` -- this is M5's real
    implementation (see README's Methods table), load-bearing for most of
    `simulation/`'s active benchmark scripts, not a one-off diagnostic.
    """
    X0=np.asarray(X,float); V=np.asarray(V,float); current=X0.copy()
    k=min(int(k),len(X0)-1)
    candidates=NearestNeighbors(n_neighbors=k+1).fit(X0).kneighbors(X0,return_distance=False)
    neighbors=np.array([row[row!=i][:k] for i,row in enumerate(candidates)])
    trajectory=[]
    def state():
        Xj=current[neighbors]; diff=Xj-current[:,None,:]; dist=np.linalg.norm(diff,axis=2)
        h=np.max(dist,axis=1)+eps; scaled=dist/h[:,None]
        w=np.maximum(0,1-scaled**2)**beta+eps; w/=w.sum(axis=1,keepdims=True)
        P=np.empty((len(current),current.shape[1],current.shape[1])); xbar=np.sum(w[:,:,None]*Xj,axis=1)
        for i in range(len(current)):
            z=Xj[i]-xbar[i]; C=(w[i,:,None]*z).T@z; _,U=np.linalg.eigh((C+C.T)/2); U=U[:,-d:]; P[i]=U@U.T
        Vp=np.einsum("nij,nj->ni",P,V)
        return P,Vp,xbar,h
    P,Vp,xbar,h=state(); trajectory.append((0,current.copy(),Vp.copy()))
    for t in range(1,T+1):
        shift=xbar-current; normal=shift-np.einsum("nij,nj->ni",P,shift); steps=eta_g*normal
        norm_=np.linalg.norm(steps,axis=1); scale=np.minimum(1,max_step_frac*h/(norm_+eps)); current=current+steps*scale[:,None]
        P,Vp,xbar,h=state(); trajectory.append((t,current.copy(),Vp.copy()))
    return trajectory

def fit_vmf_variant(Y,field,d,cfg,seed):
 """Run the existing VMF, with narrowly scoped diagnostics requested by the addendum."""
 k=min(int(cfg.get("k",50)),len(Y)-1);warm=int(cfg.get("warm_start_steps",0));start=np.asarray(Y,float)
 if warm:
  tr=position_only_trajectory(start,field,d,k,warm,cfg.get("warm_eta",cfg["eta_g"]));start=tr[-1][1]
 compatibility=np.asarray(field,float);confidence=np.ones(len(start))
 if cfg.get("compatibility_mode")=="tangent_projected":
  _,compatibility,_=local_pca_denoise(start,d,k,vectors=field,return_info=True);speed=np.linalg.norm(compatibility,axis=1);scale=np.median(speed[speed>EPS]) if np.any(speed>EPS) else 1.;confidence=np.clip(speed/(scale+EPS),0.,1.)
 vmf_T=max(1,int(cfg["T"])-warm) if warm else int(cfg["T"])
 f=VelocityManifoldFitter(start,compatibility,d_mode="global",global_d=d,k=k,T=vmf_T,eta_g=cfg["eta_g"],theta=cfg["theta"],kappa=cfg.get("kappa",1.),bandwidth_mode=cfg.get("bandwidth_mode","variable"),h=cfg.get("h",1.0),use_PCA=False,velocity_confidence=confidence,candidate_mult=cfg.get("candidate_mult",4),random_state=seed,lambda_v=cfg.get("lambda_v",0.0),velocity_covariance_mode=cfg.get("velocity_covariance_mode","centered"),velocity_trace_normalization=cfg.get("velocity_trace_normalization","match_position_trace"),record_tangent_diagnostics=cfg.get("record_tangent_diagnostics",False),return_tangent_diagnostics=cfg.get("return_tangent_diagnostics",False))
 custom=cfg.get("theta_schedule")=="ramp" or "ess_fallback_fraction" in cfg
 if not custom:
  r=f.fit(update_mode="normal_only",return_dict=True);r["V"]=np.einsum("nij,nj->ni",r["P"],field);r["effective_sample_size"]=effective_sample_size(r["weights"]);return r
 working=compatibility.copy();f._build_neighbors(working);ess_history=[];theta_final=float(cfg["theta"])
 for t in range(vmf_T):
  if cfg.get("theta_schedule")=="ramp":
   frac=(t+1)/vmf_T;f.theta=theta_final*(.2+.8*frac);f._build_neighbors(working)
  f._update_weights(velocity_mode="projected");raw=f.weights.copy()
  if "ess_fallback_fraction" in cfg:
   Xj=f.X[f.neighbors];dist=np.linalg.norm(Xj-f.X[:,None,:],axis=2);h=np.max(dist,axis=1)+EPS;spatial=np.maximum(0.,1.-(dist/h[:,None])**2)**f.beta+EPS;spatial/=spatial.sum(1,keepdims=True);ess=effective_sample_size(raw);target=float(cfg["ess_fallback_fraction"])*f.k;alpha=np.clip((ess-.5*target)/(.5*target+EPS),0.,1.);f.weights=alpha[:,None]*raw+(1-alpha[:,None])*spatial;f.weights/=f.weights.sum(1,keepdims=True)
  ess_history.append(effective_sample_size(f.weights));f._compute_local_tangent(diagnostic_iteration=t,diagnostic_phase="pre_update");f._project_velocity(working);_,shift=f._local_mean_shift();normal=shift-np.einsum("nij,nj->ni",f.P,shift);steps=f._cap_steps(f.eta_g*normal);f.X=f.X+steps;working=f.v.copy();f.history.append({"iteration":t,"mean_step_norm":float(np.mean(np.linalg.norm(steps,axis=1))),"max_step_norm":float(np.max(np.linalg.norm(steps,axis=1)))})
 f.theta=theta_final;f._update_weights(velocity_mode="projected")
 if "ess_fallback_fraction" in cfg:
  Xj=f.X[f.neighbors];dist=np.linalg.norm(Xj-f.X[:,None,:],axis=2);h=np.max(dist,axis=1)+EPS;spatial=np.maximum(0.,1.-(dist/h[:,None])**2)**f.beta+EPS;spatial/=spatial.sum(1,keepdims=True);ess=effective_sample_size(f.weights);target=float(cfg["ess_fallback_fraction"])*f.k;alpha=np.clip((ess-.5*target)/(.5*target+EPS),0.,1.);f.weights=alpha[:,None]*f.weights+(1-alpha[:,None])*spatial;f.weights/=f.weights.sum(1,keepdims=True)
 f._compute_local_tangent(diagnostic_iteration=vmf_T,diagnostic_phase="final");f._project_velocity(field);return {"X":f.X,"V":f.v,"neighbors":f.neighbors,"weights":f.weights,"P":f.P,"U":f.U,"bandwidths":f.bandwidths,"history":f.history,"effective_sample_size":effective_sample_size(f.weights),"effective_sample_size_history":ess_history,"tangent_diagnostics":f.last_tangent_diagnostics,"tangent_diagnostics_history":f.tangent_diagnostics_history,"algorithm_settings":{"lambda_v":f.lambda_v,"velocity_covariance_mode":f.velocity_covariance_mode,"velocity_trace_normalization":f.velocity_trace_normalization,"d_mode":f.d_mode,"global_d":f.global_d,"k":f.k,"T":f.T}}

def geom_fit(data,method,cfg,seed,field=None):
 Y=data["Y"];d=data["d"];k=min(int(cfg.get("k",50)),len(Y)-1)
 if method=="noisy_input":return Y.copy(),field.copy() if field is not None else None,{}
 if method=="relaxed_global_pca":X,info=relaxed_pca(Y,d,cfg["eta_pca"],cfg["T_pca"]);return X,project_vectors_with_pca_info(field,info) if field is not None else None,info
 if method=="local_pca":
  if field is None:X,info=local_pca_denoise(Y,d,k,return_info=True);return X,None,info
  X,V,info=local_pca_denoise(Y,d,k,vectors=field,return_info=True);return X,V,info
 if method=="position_only_manfit":tr=position_only_trajectory(Y,field if field is not None else np.zeros_like(Y),d,k,cfg["T"],cfg["eta_g"]);_,X,V=tr[-1];return X,V if field is not None else None,{}
 if method=="vmf":
  r=fit_vmf_variant(Y,field,d,cfg,seed);return r["X"],r["V"],r
 raise ValueError(method)

def evaluate(name,seed,method,cfg,kind,evaluator=None,n=None):
 data=vector_data(name,seed,n) if kind=="vector" else scalar_data(name,seed,n);field=data["field"] if kind=="vector" else None;X,E,info=geom_fit(data,method,cfg,seed,field);man,clean=geometry(name,X,data["P"])
 if kind=="scalar":E=common_gradient(X,data["scalar"],evaluator["k"],evaluator["ridge"])
 nrmse,cos,ret=field_metrics(E,data["truth"]);ess=np.asarray(info.get("effective_sample_size",[]),float) if isinstance(info,dict) else np.array([]);return {"information_type":kind,"scenario":name,"seed":seed,"method":method,"config_json":json.dumps(cfg,sort_keys=True),"n":len(X),"k":cfg.get("k",np.nan),"k_fraction":cfg.get("k",np.nan)/len(X) if "k" in cfg else np.nan,"distance_to_manifold":man,"clean_point_rmse":clean,"field_nrmse":nrmse,"signed_cosine":cos,"retained_fraction":ret,"median_effective_sample_size":float(np.median(ess)) if ess.size else np.nan,"min_effective_sample_size":float(np.min(ess)) if ess.size else np.nan}

RELAX=[{"eta_pca":e,"T_pca":T} for e in (.2,.35,.5,.7,1.) for T in (1,3,5,8)]
def candidates(method,kbest=None,scalar=False):
 if method=="noisy_input":return [{}]
 if method=="relaxed_global_pca":return RELAX
 if method=="local_pca":return [{"k":k} for k in KGRID]
 if method=="position_only_manfit":return ([{"k":k,"eta_g":.35,"T":5} for k in KGRID] if kbest is None else [{"k":kbest,"eta_g":e,"T":T} for e in ETA for T in TGRID])
 if method=="vmf":return ([{"k":k,"theta":.1,"eta_g":.35,"T":5,"kappa":1.} for k in KGRID] if kbest is None else [{"k":kbest,"theta":th,"eta_g":e,"T":T} for th in THETA for e in ETA for T in TGRID])

def score_table(rows):
 d=pd.DataFrame(rows);q=d.groupby("config_json")[["distance_to_manifold","clean_point_rmse","field_nrmse","signed_cosine"]].median().reset_index();return q
def best_geometry(rows):q=score_table(rows);return json.loads(q.sort_values(["clean_point_rmse","distance_to_manifold"]).iloc[0].config_json)
def best_field(rows,pos_best_clean):
 q=score_table(rows);feasible=q[q.clean_point_rmse<=1.1*pos_best_clean]
 if not len(feasible):raise RuntimeError(f"no field-informed configuration satisfies the geometry safeguard: best clean RMSE={q.clean_point_rmse.min():.6g}, limit={1.1*pos_best_clean:.6g}")
 return json.loads(feasible.sort_values(["field_nrmse","signed_cosine","distance_to_manifold"],ascending=[True,False,True]).iloc[0].config_json)

def tune_scenario(name,kind,seeds):
 methods=VMETHODS if kind=="vector" else SMETHODS;evaluator={"k":50,"ridge":.05};scalar_scale=1.
 if kind=="scalar":
  er=[]
  for k in (30,50,80):
   for ridge in (.005,.02,.05,.1):
    for seed in seeds:
     data=scalar_data(name,seed);E=common_gradient(data["Y"],data["scalar"],k,ridge);nr,co,_=field_metrics(E,data["truth"]);er.append({"k":k,"ridge":ridge,"nrmse":nr,"cos":co})
  evaluator=pd.DataFrame(er).groupby(["k","ridge"])["nrmse"].median().idxmin();evaluator={"k":int(evaluator[0]),"ridge":float(evaluator[1])};scalar_scale=np.median([np.median(abs(scalar_data(name,z)["scalar"]-np.median(scalar_data(name,z)["scalar"]))) for z in seeds])
 allrows=[];selected={"evaluator":evaluator}
 for method in methods:
  if method=="noisy_input":selected[method]={};continue
  stage=[]
  for cfg in candidates(method):
   cfg=dict(cfg)
   if method=="scalar_potential_manfit":cfg["h_s"]*=scalar_scale
   for seed in seeds:stage.append(evaluate(name,seed,method,cfg,kind,evaluator))
  allrows+=stage;kbest=best_geometry(stage).get("k") if method not in ("vmf","scalar_potential_manfit") else best_field(stage,np.inf).get("k")
  if method=="vmf":
   wide=[]
   for k_choice in sorted({int(kbest),int(selected["position_only_manfit"]["k"])}):
    compatibility=[]
    for kappa in KAPPA:
     cfg={"k":k_choice,"theta":.1,"eta_g":.35,"T":5,"kappa":kappa}
     for seed in seeds:compatibility.append(evaluate(name,seed,method,cfg,kind,evaluator))
    allrows+=compatibility;kappa_best=best_field(compatibility,np.inf)["kappa"]
    for cfg in candidates(method,k_choice):
     cfg=dict(cfg);cfg["kappa"]=kappa_best
     for seed in seeds:wide.append(evaluate(name,seed,method,cfg,kind,evaluator))
   allrows+=wide;stage=wide
  elif method=="scalar_potential_manfit":
   wide=[]
   for k_choice in sorted({int(kbest),int(selected["position_only_manfit"]["k"])}):
    bandwidth=[]
    for h in (.5,1.,2.,4.,8.):
     cfg={"k":k_choice,"eta":.5,"T":5,"h_s":h*scalar_scale}
     for seed in seeds:bandwidth.append(evaluate(name,seed,method,cfg,kind,evaluator))
    allrows+=bandwidth;h_best=best_field(bandwidth,np.inf)["h_s"]
    for e in ETA:
     for T in TGRID:
      cfg={"k":k_choice,"eta":e,"T":T,"h_s":h_best}
      for seed in seeds:wide.append(evaluate(name,seed,method,cfg,kind,evaluator))
   allrows+=wide;stage=wide
  elif method=="position_only_manfit":
   wide=[]
   for cfg in candidates(method,kbest):
    cfg=dict(cfg)
    for seed in seeds:wide.append(evaluate(name,seed,method,cfg,kind,evaluator))
   allrows+=wide;stage=wide
  if method in ("vmf","scalar_potential_manfit"):
   pos_clean=np.inf;selected[method]=best_field(stage,pos_clean)
  else:selected[method]=best_geometry(stage)
 # Apply the explicit geometry safeguard after position-only selection.
 posrows=[r for r in allrows if r["method"]=="position_only_manfit" and json.loads(r["config_json"])==selected["position_only_manfit"]];posclean=pd.DataFrame(posrows).clean_point_rmse.median()
 for method in (("vmf",) if kind=="vector" else ("scalar_potential_manfit",)):
  rows=[r for r in allrows if r["method"]==method];selected[method]=best_field(rows,posclean)
 return selected,allrows

def add_relative(d):
 keys=["information_type","scenario","seed"];raw=d[d.method=="noisy_input"].set_index(keys)
 for col in ("distance_to_manifold","clean_point_rmse","field_nrmse"):d[col+"_rel"]=(d[col]+EPS)/(np.asarray(d.set_index(keys).index.map(raw[col]))+EPS)
 d["signed_cosine_improvement"]=d.signed_cosine-np.asarray(d.set_index(keys).index.map(raw.signed_cosine));return d

def sample_study(selected):
 rows=[]
 for kind,name in (("vector","s_curve"),("scalar","scalar_s_curve")):
  n0=SETS[name].n
  for n in (n0,2*n0,4*n0):
   for regime in ("fixed_absolute","tuned_absolute"):
    ks=(30,50,80) if regime=="fixed_absolute" else (20,30,50,80,120)
    ev=selected[name]["evaluator"];target="vmf" if kind=="vector" else "scalar_potential_manfit"
    for seed in range(34000,34003):
     r=evaluate(name,seed,"noisy_input",{},kind,ev,n);r["regime"]=regime;r["study_k"]=np.nan;rows.append(r)
    for k in ks:
     for seed in range(34000,34003):
      for method in ("position_only_manfit",target):
       cfg=dict(selected[name][method]);cfg["k"]=k;r=evaluate(name,seed,method,cfg,kind,ev,n);r["regime"]=regime;r["study_k"]=k;rows.append(r)
 d=pd.DataFrame(rows);d["selected_for_n"]=False;d["geometry_feasible"]=pd.Series(pd.NA,index=d.index,dtype="boolean")
 for kind,name in (("vector","s_curve"),("scalar","scalar_s_curve")):
  target="vmf" if kind=="vector" else "scalar_potential_manfit"
  for n in sorted(d[d.scenario.eq(name)].n.unique()):
   z=d[(d.scenario.eq(name))&(d.n.eq(n))&(d.regime.eq("tuned_absolute"))]
   pos=z[z.method.eq("position_only_manfit")].groupby("study_k").clean_point_rmse.median();pos_k=int(pos.idxmin());limit=1.1*pos.min();field=z[z.method.eq(target)].groupby("study_k").agg(clean=("clean_point_rmse","median"),field=("field_nrmse","median"),cos=("signed_cosine","median"),man=("distance_to_manifold","median"));feasible=field[field.clean<=limit]
   field_mask=(d.scenario.eq(name))&(d.n.eq(n))&(d.regime.eq("tuned_absolute"))&d.method.eq(target);d.loc[field_mask,"geometry_feasible"]=d.loc[field_mask,"study_k"].isin(feasible.index)
   field_k=None if not len(feasible) else int(feasible.sort_values(["field","cos","man"],ascending=[True,False,True]).index[0]);chosen_field=False if field_k is None else ((d.method.eq(target))&(d.study_k.eq(field_k)));mask=(d.scenario.eq(name))&(d.n.eq(n))&(d.regime.eq("tuned_absolute"))&(((d.method.eq("position_only_manfit"))&(d.study_k.eq(pos_k)))|chosen_field|d.method.eq("noisy_input"));d.loc[mask,"selected_for_n"]=True
 return d

def image_uri(path):return "data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()
def representative(final,name,kind,selected,out):
 target="vmf" if kind=="vector" else "scalar_potential_manfit";z=final[(final.scenario==name)&(final.method==target)].copy();metrics=["distance_to_manifold","clean_point_rmse","field_nrmse","signed_cosine"];center=z[metrics].median();scale=(z[metrics].quantile(.75)-z[metrics].quantile(.25)).replace(0,1);distance=((z[metrics]-center)/scale).abs().sum(1);seed=int(z.loc[distance.idxmin(),"seed"]);data=vector_data(name,seed) if kind=="vector" else scalar_data(name,seed);methods=VMETHODS if kind=="vector" else SMETHODS;fig,axes=plt.subplots(2,3,figsize=(12,7),subplot_kw={"projection":"3d"} if name in ("flat_rotation_annulus","half_sphere_tangent","scalar_saddle") else {},constrained_layout=True);pan=[]
 for method in methods:
  cfg=selected[name][method];field=data.get("field") if kind=="vector" else None;X,E,_=geom_fit(data,method,cfg,seed,field)
  if kind=="scalar":E=common_gradient(X,data["scalar"],selected[name]["evaluator"]["k"],selected[name]["evaluator"]["ridge"])
  pan.append((method,X,E))
 for ax,(method,X,E) in zip(axes.ravel(),pan):
  ax.scatter(X[:,0],X[:,1],X[:,2] if hasattr(ax,"zaxis") else None,s=6,alpha=.6) if hasattr(ax,"zaxis") else ax.scatter(X[:,0],X[:,1],s=6,alpha=.6)
  step=max(1,len(X)//35)
  if hasattr(ax,"zaxis"):ax.quiver(X[::step,0],X[::step,1],X[::step,2],E[::step,0],E[::step,1],E[::step,2],length=.12,normalize=True)
  else:ax.quiver(X[::step,0],X[::step,1],E[::step,0],E[::step,1],scale=18)
  if hasattr(ax,"zaxis"):ax.view_init(elev=24,azim=-58)
  ax.set_title(LABEL[method]);ax.set_xticks([]);ax.set_yticks([]);getattr(ax,"set_zticks",lambda x:None)([])
 for ax in axes.ravel()[len(pan):]:fig.delaxes(ax)
 p=out/f"representative_{name}.png";fig.suptitle(f"{name.replace('_',' ').title()} — componentwise-median seed {seed}");fig.savefig(p,dpi=185,bbox_inches="tight",facecolor="white");plt.close(fig);return p,seed

def boxplot(final,name,kind,out):
 methods=VMETHODS if kind=="vector" else SMETHODS;z=final[(final.scenario==name)&(final.method!="noisy_input")];metrics=("distance_to_manifold_rel","clean_point_rmse_rel","field_nrmse_rel","signed_cosine_improvement");titles=("Relative manifold distance","Relative clean-point RMSE",("Relative velocity NRMSE" if kind=="vector" else "Relative gradient NRMSE"),("Signed velocity-cosine improvement" if kind=="vector" else "Signed gradient-cosine improvement"));fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
 for ax,m,t in zip(axes.ravel(),metrics,titles):g=[z[z.method==x][m] for x in methods[1:]];ax.boxplot(g,tick_labels=[LABEL[x] for x in methods[1:]],showfliers=False);ax.axhline(0 if m=="signed_cosine_improvement" else 1,color="black",ls="--");ax.set_title(t);ax.tick_params(axis="x",rotation=25,labelsize=7)
 p=out/f"relative_{name}.png";fig.savefig(p,dpi=185,bbox_inches="tight",facecolor="white");plt.close(fig);return p

def build_report(out,final,summary,selected,vector_scenarios=None):
 vector_scenarios=VECTOR if vector_scenarios is None else tuple(vector_scenarios);figs=out/"figures";figs.mkdir(exist_ok=True);style="body{margin:0;background:#f4f6f8;color:#1f2933;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1050px;margin:auto;padding:30px 20px 60px}.card{background:white;border:1px solid #d8dde6;border-radius:8px;padding:20px;margin:18px 0}h2{border-bottom:1px solid #d8dde6;padding-bottom:8px}p{line-height:1.5}img{max-width:100%;border:1px solid #d8dde6;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d8dde6;padding:7px}th{background:#edf1f5}.small{font-size:12px;color:#52606d}";parts=[f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Simulation</title><style>{style}</style></head><body><main><h1>Simulation</h1><p>Vector-field and scalar-function experiments</p><section class='card'><p>Relative error metrics are scaled by the corresponding noisy-input error within the same scenario and seed. A value of 1 indicates no improvement over the noisy baseline, while values below 1 indicate lower error.</p><p>Geometry uses RMS distance to the true support and fixed-target clean-point RMSE. Field recovery uses energy-normalized RMSE and signed cosine consistency.</p></section>"]
 # Part II (scalar functions) removed 2026-08-12 along with scalar_potential_manfit;
 # see simulation/current_plan.md P4.0.
 for heading,names,kind in (("Part I. Manifold fitting with vector fields",vector_scenarios,"vector"),):
  parts.append(f"<h2>{heading}</h2>")
  for name in names:
   rep,seed=representative(final,name,kind,selected,figs);box=boxplot(final,name,kind,figs);methods=VMETHODS if kind=="vector" else SMETHODS;q=summary[(summary.scenario==name)&summary.method.isin(methods)].set_index("method").reindex(methods).reset_index();q.method=q.method.map(LABEL);cols=["method","distance_to_manifold","clean_point_rmse","field_nrmse","signed_cosine"];q=q[cols].rename(columns={"field_nrmse":"velocity NRMSE" if kind=="vector" else "gradient NRMSE","signed_cosine":"signed cosine" if kind=="vector" else "signed gradient cosine","distance_to_manifold":"distance to manifold","clean_point_rmse":"clean-point RMSE"})
   parts.append(f"<section class='card'><h2>{name.replace('_',' ').title()}</h2><p>{SETUP[name]}</p><p class='small'>n={SETS[name].n}; position noise={SETS[name].px}; observed field/scalar noise={SETS[name].field_noise}</p><h3>Representative method comparison</h3><img src='{image_uri(rep)}'><p class='small'>Final seed {seed}, closest to componentwise median method performance.</p><h3>Median final metrics</h3>{q.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')}<h3>Seed-level relative metrics</h3><img src='{image_uri(box)}'><p class='small'>Errors are relative to noisy input (1 = no improvement); signed cosine is shown as improvement over noisy input (0 = no improvement).</p></section>")
 parts.append("</main></body></html>");(out/"final_report.html").write_text("".join(parts))

def checks():
 # Vector metric checks.
 T=np.array([[1.,0.,0.],[0.,1.,0.]]);perfect=field_metrics(T,T);double=field_metrics(2*T,T);opp=field_metrics(-T,T);permuted=field_metrics(T[::-1],T);assert perfect[0]<1e-12 and abs(perfect[1]-1)<1e-10 and double[0]>0 and abs(double[1]-1)<1e-10 and abs(opp[1]+1)<1e-10 and permuted[0]>perfect[0]
 Y=np.array([[-1.,0.,.1],[0.,0.,-.1],[1.,0.,.2]]);V=np.array([[1.,2.,3.],[3.,2.,1.],[-1.,2.,4.]]);Xp,info=relaxed_pca(Y,1,.35,3);Vp=project_vectors_with_pca_info(V,info);assert Xp.shape==Y.shape and Vp.shape==V.shape and np.all(np.isfinite(Vp))
 branch_targets=np.array([[0.,-.1,0.],[0.,.1,0.]]);switched=branch_targets[::-1];branch_clean=float(np.sqrt(np.mean(np.sum((switched-branch_targets)**2,axis=1))));branch_rematch=float(np.sqrt(np.mean(NearestNeighbors(n_neighbors=1).fit(branch_targets).kneighbors(switched)[0][:,0]**2)));assert branch_clean>.1 and branch_rematch==0
 # Scalar evaluator invariance and plane linear-gradient correctness.
 rng=np.random.default_rng(2);X=np.c_[rng.normal(size=(100,2)),np.zeros(100)];S=2*X[:,0]-X[:,1];g=common_gradient(X,S,40,1e-3);g2=common_gradient(X,S+7,40,1e-3);assert np.allclose(g,g2) and np.mean(np.linalg.norm(g-np.array([2.,-1.,0.]),axis=1))<.1
 d=scalar_data("scalar_saddle",3);assert np.max(abs((d["truth"]*norm(np.cross(np.c_[np.ones(len(d['P'])),np.zeros(len(d['P'])),.9*d['P'][:,0]],np.c_[np.zeros(len(d['P'])),np.ones(len(d['P'])),-.9*d['P'][:,1]]))).sum(1)))<1e-8
 # scalar_potential_manfit (fit_potential_aware_neighborhoods) removed 2026-08-12,
 # see simulation/current_plan.md P4.0 -- superseded by scripts.scalar_potential_manfit
 # .fit_scalar_gradient_manfit, which is not wired into this legacy report's
 # SMETHODS/geom_fit. The two checks that used to live here (its signature and a
 # truth-freeness invariance check) were removed along with it.
 return {"perfect_nrmse":perfect[0],"perfect_cosine":perfect[1],"doubled_nrmse":double[0],"doubled_cosine":double[1],"opposite_cosine":opp[1],"permuted_nrmse":permuted[0],"pca_velocity_ambient_shape":list(Vp.shape),"branch_switched_fixed_target_rmse":branch_clean,"branch_switched_rematched_rmse":branch_rematch,"scalar_constant_shift_invariant":True,"plane_linear_gradient_correct":True,"saddle_intrinsic_gradient_tangent":True}

def main():
 a=argparse.ArgumentParser();a.add_argument("--output-dir",type=Path,default=ROOT/"results/field_informed_manfit_benchmark");a.add_argument("--report-only",action="store_true",help="rebuild figures and HTML from completed local CSV/configuration files");x=a.parse_args();out=x.output_dir
 if x.report_only:
  selected={p.stem:json.loads(p.read_text()) for p in (out/"selected_configs").glob("*.json")};fd=pd.read_csv(out/"final_results_long.csv");fs=pd.read_csv(out/"final_results_summary.csv");scenario_file=out/"report_scenarios.json";vector_scenarios=json.loads(scenario_file.read_text())["vector"] if scenario_file.exists() else VECTOR;build_report(out,fd,fs,selected,vector_scenarios);print(out/"final_report.html");return
 out.mkdir(parents=True,exist_ok=False);(out/"selected_configs").mkdir();audit=checks();(out/"metric_and_method_audit.md").write_text("# Metric and method audit\n\n- The audited full PCA baseline fits one centered rank-d model, reconstructs every position in that fixed affine subspace, and projects observed velocities through the same components in ambient coordinates. It remains available as `full_global_pca_projection` but is not a main report row.\n- The finite global-PCA row uses the same fixed affine model and returns `projection + (1 - eta_PCA)^T_PCA * (observed - projection)`. Its reported velocity remains the fully tangent-projected PCA velocity.\n- VMF tuning covers `theta`, `eta_g`, `T`, `k`, and the directional compatibility scale `kappa`; the 10% clean-RMSE safeguard is a hard constraint. Targeted addendum diagnostics also record effective sample size for the bounded A-E ablations.\n- scalar_potential_manfit (fit_potential_aware_neighborhoods) was removed 2026-08-12; see simulation/current_plan.md P4.0. This report now covers vector scenarios only.\n- Distance to manifold uses analytic distances where available and a seed-independent dense support otherwise. Fixed clean targets preserve sample and branch/component identity; no target rematching is performed. Signed cosine retains sign and excludes only near-zero true fields under one common threshold.\n\n## Deterministic checks\n\n```json\n"+json.dumps(audit,indent=2)+"\n```\n")
 # Separate pilot seeds verify that fixed difficulties are nondegenerate before tuning.
 # kind="scalar" removed 2026-08-12 along with scalar_potential_manfit (see
 # simulation/current_plan.md P4.0) -- tune_scenario/representative/sample_study still
 # have internal scalar-specific branches, but they hardcode "scalar_potential_manfit"
 # as the representative scalar method and would break if ever driven with kind=
 # "scalar" again now that it's gone; not exercised because this loop no longer does.
 pilot=[]
 for kind,names,methods in (("vector",VECTOR,VMETHODS),):
  for name in names:
   ev={"k":50,"ridge":.05}
   for seed in range(41000,41002):
    for method in methods:pilot.append(evaluate(name,seed,method,({} if method=="noisy_input" else (RELAX[6] if method=="relaxed_global_pca" else ({"k":50} if method=="local_pca" else ({"k":50,"eta_g":.35,"T":5,"theta":.1} if method=="vmf" else {"k":50,"eta_g":.35,"T":5})))),kind,ev))
 pd.DataFrame(pilot).to_csv(out/"pilot_results_long.csv",index=False)
 tuning=[];selected={}
 for kind,names in (("vector",VECTOR),):
  for name in names:
   sel,rows=tune_scenario(name,kind,range(42000,42003));selected[name]=sel;tuning+=rows;(out/"selected_configs"/f"{name}.json").write_text(json.dumps(sel,indent=2)+"\n")
 td=pd.DataFrame(tuning);td.to_csv(out/"tuning_results_long.csv",index=False)
 final=[]
 for kind,names,methods in (("vector",VECTOR,VMETHODS),):
  for name in names:
   for seed in range(43000,43015):
    for method in methods:final.append(evaluate(name,seed,method,selected[name][method],kind,selected[name]["evaluator"]))
 fd=add_relative(pd.DataFrame(final));fd.to_csv(out/"final_results_long.csv",index=False);fs=fd.groupby(["information_type","scenario","method"])[["distance_to_manifold","clean_point_rmse","field_nrmse","signed_cosine"]].median().reset_index();fs.to_csv(out/"final_results_summary.csv",index=False);sample_study(selected).to_csv(out/"sample_size_k_study.csv",index=False);build_report(out,fd,fs,selected);print(out/"final_report.html")
if __name__=="__main__":main()
