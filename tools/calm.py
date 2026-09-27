# Calm single-move edit of the faithful push-in clip:
# slowed dolly-in framed on the TV side, then ONE slow glide right to the sofa/pool, easing to a near stop.
import numpy as np, subprocess, sys
from PIL import Image, ImageFilter
FF=sys.argv[1]; HI=192; fps=24; W,H=752,560; OW,OH=720,1280; ww=H*OW/OH; xmax=W-ww
DUR=14.8; N=int(round(DUR*fps)); SRC=8.0
# speed profile (output time -> source speed), integrated to source time
tt=np.linspace(0,DUR,40001); base=0.8; t1=7.0; T=6.85
s=np.where(tt<t1,base,0.06+(base-0.06)*np.clip(1-(tt-t1)/T,0,1)**1.5)
s*=np.clip(tt/0.8,0,1)**0.5*0.0+1  # constant start (camera already gliding)
src=np.concatenate([[0],np.cumsum(s[:-1]*np.diff(tt))]); src=np.minimum(src,SRC-0.05)
def sm(u): u=np.clip(u,0,1); return u*u*u*(u*(6*u-15)+10)
dec=subprocess.Popen([FF,'-v','error','-i','one43.mp4','-vf',f'minterpolate=fps={HI}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','13','-preset','slow','-pix_fmt','yuv420p','calm916.mp4'],stdin=subprocess.PIPE)
cur=-1; buf=None
for k in range(N):
    t=k/fps; want=int(round(np.interp(t,tt,src)*HI))
    while cur<want:
        b=dec.stdout.read(W*H*3)
        if len(b)<W*H*3: break
        buf=b; cur+=1
    x=xmax*0.85*sm((t-4.0)/6.5)
    fr=Image.frombuffer('RGB',(W,H),buf,'raw','RGB',0,1).resize((OW,OH),Image.LANCZOS,box=(x,0,x+ww,H))
    enc.stdin.write(fr.filter(ImageFilter.UnsharpMask(radius=1.5,percent=40,threshold=2)).tobytes())
enc.stdin.close(); enc.wait(); dec.kill(); print('frames',N,'last src frame',cur)
