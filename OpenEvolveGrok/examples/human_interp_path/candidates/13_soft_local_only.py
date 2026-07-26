
import numpy as np
# EVOLVE-BLOCK-START
def _so3_log(R):
    cos_t=float(np.clip((np.trace(R)-1)*0.5,-1,1)); th=float(np.arccos(cos_t))
    if th<1e-8: return np.zeros(3)
    w=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    return w*(th/(2*np.sin(th)+1e-12))
def _so3_exp(w):
    th=float(np.linalg.norm(w))
    if th<1e-10: return np.eye(3)
    k=w/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0.]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
def predict_interior(ca_start, ca_end, fracs, mask):
    ca0=np.asarray(ca_start,float); ca1=np.asarray(ca_end,float); m=np.asarray(mask,bool)
    L=ca0.shape[0]; half=13; out=np.empty((len(fracs),L,3))
    disp=np.linalg.norm(ca1-ca0,axis=1)
    for i in range(L):
        lo,hi=max(0,i-half),min(L,i+half+1)
        sel=np.arange(lo,hi); sel=sel[m[sel]]
        if sel.size<3:
            for t,s in enumerate(fracs): out[t,i]=(1-s)*ca0[i]+s*ca1[i]
            continue
        ww=np.exp(-0.5*((sel-i)/max(half/2,1.))**2)*(0.2+disp[sel]); ww/=ww.sum()+1e-12
        mu0=(ww[:,None]*ca0[sel]).sum(0); mu1=(ww[:,None]*ca1[sel]).sum(0)
        X=(ca0[sel]-mu0)*ww[:,None]; Y=ca1[sel]-mu1; H=X.T@Y
        try:
            U,S,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T)) or 1.
            R=Vt.T@np.diag([1.,1.,d])@U.T
        except Exception:
            R=np.eye(3)
        om=_so3_log(R); res=ca1[i]-((ca0[i]-mu0)@R.T+mu1)
        for t,s in enumerate(fracs):
            s=float(s); Rs=_so3_exp(om*s); mus=(1-s)*mu0+s*mu1
            out[t,i]=(ca0[i]-mu0)@Rs.T+mus+s*res
    return out
# EVOLVE-BLOCK-END
def run_interpolation(ca_start,ca_end,fracs,mask):
    return predict_interior(ca_start,ca_end,fracs,mask)
