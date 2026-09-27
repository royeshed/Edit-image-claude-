# Vertical 9:16 window gliding left->right across the 4:3 clip (TV side -> doors -> sofa/artworks).
import numpy as np, subprocess, sys
from PIL import Image, ImageFilter
FF=sys.argv[1]; W,H=752,560; OW,OH=720,1280; fps=24
ww=H*OW/OH  # 315 px window
dec=subprocess.Popen([FF,'-v','error','-i','base43.mp4','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p','pan916.mp4'],stdin=subprocess.PIPE)
x0,x1=0.0,(W-ww)*0.80; t0,t1=0.4,9.6
k=0
while True:
    buf=dec.stdout.read(W*H*3)
    if len(buf)<W*H*3: break
    t=k/fps; u=np.clip((t-t0)/(t1-t0),0,1); e=u*u*u*(u*(6*u-15)+10)  # smootherstep glide
    x=x0+(x1-x0)*e
    im=Image.frombuffer('RGB',(W,H),buf,'raw','RGB',0,1)
    fr=im.resize((OW,OH),Image.LANCZOS,box=(x,0,x+ww,H)).filter(ImageFilter.UnsharpMask(radius=1.6,percent=45,threshold=2))
    enc.stdin.write(fr.tobytes()); k+=1
enc.stdin.close(); enc.wait(); print('frames',k)
