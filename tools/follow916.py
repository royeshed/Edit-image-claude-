# 9:16 window that follows the camera: left->right during the truck, back to centre during the pull-back, centred for the pool push-in.
import numpy as np, subprocess, sys
from PIL import Image, ImageFilter
from scipy.interpolate import PchipInterpolator
FF=sys.argv[1]; src=sys.argv[2]; out=sys.argv[3]
W,H=752,560; OW,OH,fps=720,1280,24; ww=H*OW/OH; xmax=W-ww
kx=PchipInterpolator([0,0.3,3.9,7.9,99],[0,0,xmax,xmax/2,xmax/2])
dec=subprocess.Popen([FF,'-v','error','-i',src,'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','13','-preset','slow','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
k=0
while True:
    b=dec.stdout.read(W*H*3)
    if len(b)<W*H*3: break
    x=float(np.clip(kx(k/fps),0,xmax))
    fr=Image.frombuffer('RGB',(W,H),b,'raw','RGB',0,1).resize((OW,OH),Image.LANCZOS,box=(x,0,x+ww,H))
    enc.stdin.write(fr.filter(ImageFilter.UnsharpMask(radius=1.5,percent=40,threshold=2)).tobytes()); k+=1
enc.stdin.close(); enc.wait(); print('frames',k)
