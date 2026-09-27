# Calm retime of the chained 4:3 master: gentle constant pace, eased slow-downs at each move change,
# long gradual deceleration into the pool, frame-interpolated so every step is smooth.
import numpy as np, subprocess, sys
from PIL import Image
FF=sys.argv[1]; HI=192;
# de-judder map: real time -> fractional frame position (the generator skips ~1 in 5 frames, so every 4th step is ~1.45x longer)
_b=subprocess.run([FF,'-v','error','-i','chain4.mp4','-vf','scale=160:120','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
_F=np.frombuffer(_b,np.uint8).reshape(-1,120,160).astype(float); _d=np.abs(np.diff(_F,axis=0)).mean(axis=(1,2)); _u=np.ones(len(_d))
for _i in range(len(_d)):
    _m=np.median(_d[max(0,_i-6):_i+7])
    if _d[_i]>1.35*_m: _u[_i]=1.45
TS=np.concatenate([[0],np.cumsum(_u)]); TS=TS/TS[-1]*(len(_F)-1)/24; IDX=np.arange(len(TS))
fps=24; W,H=752,560; OW,OH=960,720
SRC=float(TS[-1]); JOINS=(float(TS[96]),float(TS[191]),float(TS[286])); BASE=1.3; DEC_AT=float(TS[286])+6.2; DT=1/2400
src=0.0; t=0.0; decay_t=None; ts=[]; ss=[]; vv=[]
while True:
    d=min(abs(src-j) for j in JOINS); s=BASE*(0.5+0.5*min(1,d/0.55)**1.5)
    if src>=DEC_AT:
        decay_t=t if decay_t is None else decay_t; x=min(1,(t-decay_t)/3.2); s=0.22+(BASE-0.22)*(1-x)**2
    ts.append(t); ss.append(src); vv.append(s); src+=s*DT; t+=DT
    if src>=SRC-0.06: break
T_END=t+0.35; N=int(round(T_END*fps)); ts=np.array(ts); ss=np.array(ss); vv=np.array(vv)
print('duration %.2f  decel starts %.2f'%(T_END,decay_t))
open('natural_timing.txt','w').write('%.3f %.3f\n'%(T_END,decay_t))
dec=subprocess.Popen([FF,'-v','error','-i','chain4.mp4','-vf',f'minterpolate=fps={HI}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','13','-preset','slow','-pix_fmt','yuv420p','natural4.mp4'],stdin=subprocess.PIPE)
cur=-1; buf=None
for k in range(N):
    want=int(round(np.interp(np.interp(k/fps,ts,ss),TS,IDX)*HI/24))
    while cur<want:
        b=dec.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        buf=b; cur+=1
    tt_=k/fps; sp=float(np.interp(tt_,ts,vv))/BASE
    nx=0.55*np.sin(2*np.pi*0.23*tt_+0.7)+0.30*np.sin(2*np.pi*0.61*tt_+2.1)+0.15*np.sin(2*np.pi*1.37*tt_+4.0)
    ny=0.50*np.sin(2*np.pi*0.19*tt_+1.9)+0.30*np.sin(2*np.pi*0.53*tt_+0.3)+0.20*np.sin(2*np.pi*1.21*tt_+3.3)
    nr=0.6*np.sin(2*np.pi*0.17*tt_+0.9)+0.4*np.sin(2*np.pi*0.47*tt_+2.6)
    dx=2.2*nx; dy=1.8*ny+0.9*sp*np.sin(2*np.pi*1.75*tt_); rot=np.deg2rad(0.14*nr)
    im=Image.frombuffer('RGB',(W,H),buf,'raw','RGB',0,1).resize((OW,OH),Image.LANCZOS)
    sc=1.03; c,sn=np.cos(rot)/sc,np.sin(rot)/sc; cx,cy=OW/2,OH/2
    a,b_=c,sn; d,e=-sn,c; cc=cx-a*(cx+dx)-b_*(cy+dy); f=cy-d*(cx+dx)-e*(cy+dy)
    enc.stdin.write(im.transform((OW,OH),Image.AFFINE,(a,b_,cc,d,e,f),resample=Image.BICUBIC).tobytes())
enc.stdin.close(); enc.wait(); dec.kill(); print('frames',N,'last',cur)
