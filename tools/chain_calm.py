# Calm retime of the chained 4:3 master: gentle constant pace, eased slow-downs at each move change,
# long gradual deceleration into the pool, frame-interpolated so every step is smooth.
import numpy as np, subprocess, sys
from PIL import Image
FF=sys.argv[1]; HI=192;
# de-judder map: real time -> fractional frame position (the generator skips ~1 in 5 frames, so every 4th step is ~1.45x longer)
_b=subprocess.run([FF,'-v','error','-i','chain43.mp4','-vf','scale=160:120','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True).stdout
_F=np.frombuffer(_b,np.uint8).reshape(-1,120,160).astype(float); _d=np.abs(np.diff(_F,axis=0)).mean(axis=(1,2)); _u=np.ones(len(_d))
for _i in range(len(_d)):
    _m=np.median(_d[max(0,_i-6):_i+7])
    if _d[_i]>1.35*_m: _u[_i]=1.45
TS=np.concatenate([[0],np.cumsum(_u)]); TS=TS/TS[-1]*(len(_F)-1)/24; IDX=np.arange(len(TS))
fps=24; W,H=752,560; OW,OH=960,720
SRC=16.0; JOINS=(float(TS[96]),float(TS[192])); BASE=0.85; DEC_AT=13.9; DT=1/2400
src=0.0; t=0.0; decay_t=None; ts=[]; ss=[]
while True:
    d=min(abs(src-j) for j in JOINS); s=BASE*(0.5+0.5*min(1,d/0.7)**1.5)
    if src>=DEC_AT:
        decay_t=t if decay_t is None else decay_t; x=min(1,(t-decay_t)/5.4); s=0.05+(BASE-0.05)*(1-x)**1.5
    ts.append(t); ss.append(src); src+=s*DT; t+=DT
    if src>=SRC-0.06: break
T_END=t+0.9; N=int(round(T_END*fps)); ts=np.array(ts); ss=np.array(ss)
print('duration %.2f  decel starts %.2f'%(T_END,decay_t))
open('chain_timing.txt','w').write('%.3f %.3f\n'%(T_END,decay_t))
dec=subprocess.Popen([FF,'-v','error','-i','chain43.mp4','-vf',f'minterpolate=fps={HI}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','13','-preset','slow','-pix_fmt','yuv420p','chaincalm43_dj.mp4'],stdin=subprocess.PIPE)
cur=-1; buf=None
for k in range(N):
    want=int(round(np.interp(np.interp(k/fps,ts,ss),TS,IDX)*HI/24))
    while cur<want:
        b=dec.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        buf=b; cur+=1
    enc.stdin.write(Image.frombuffer('RGB',(W,H),buf,'raw','RGB',0,1).resize((OW,OH),Image.LANCZOS).tobytes())
enc.stdin.close(); enc.wait(); dec.kill(); print('frames',N,'last',cur)
