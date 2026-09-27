# Faithful push-in on the original photo: sub-pixel crop animation, no AI.
# Motion: ease-in, steady glide, then a long gradual deceleration (speed ramp) to a near stop.
import numpy as np, subprocess, sys
from PIL import Image
img=Image.open(sys.argv[1]).convert('RGB'); W0,H0=img.size
OW,OH,fps,dur=720,1280,24,12.29; N=int(round(dur*fps))
ar=OW/OH
# start: full-height 9:16 window centred on the doors/coffee-table axis
h0=H0; w0=h0*ar; cx0=622; cy0=H0/2
# end: ~1.75x closer, centred on the glass doors / pool
z=1.75; h1=H0/z; w1=h1*ar; cx1=632; cy1=372
t=np.arange(N)/fps
v=np.ones(N)
up=np.clip(t/1.2,0,1); v*=up*up*(3-2*up)                 # ease-in
x=np.clip((t-5.2)/(dur-5.2),0,1); v*=0.03+0.97*(1-x)**2.6   # gradual slow-down
p=np.cumsum(v); p=(p-p[0])/(p[-1]-p[0])
ff=subprocess.Popen([sys.argv[2],'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-',
    '-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p',sys.argv[3]],stdin=subprocess.PIPE)
for k in range(N):
    s=p[k]; w=w0+(w1-w0)*s; h=h0+(h1-h0)*s; cx=cx0+(cx1-cx0)*s; cy=cy0+(cy1-cy0)*s
    cx=min(max(cx,w/2),W0-w/2); cy=min(max(cy,h/2),H0-h/2)
    fr=img.resize((OW,OH),Image.LANCZOS,box=(cx-w/2,cy-h/2,cx+w/2,cy+h/2))
    ff.stdin.write(fr.tobytes())
ff.stdin.close(); ff.wait(); print('frames',N)
