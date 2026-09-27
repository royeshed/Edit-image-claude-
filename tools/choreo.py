# Camera choreography on the original photo (no AI):
# 1) slight zoom, glide left->right  2) zoom out to the full photo  3) zoom in to the pool, easing to a near stop.
import numpy as np, subprocess, sys
from PIL import Image, ImageFilter, ImageEnhance
from scipy.interpolate import PchipInterpolator
img=Image.open(sys.argv[1]).convert('RGB'); FF=sys.argv[2]; out=sys.argv[3]
W,H=img.size; OW,OH,fps=720,1280,24; ar=OW/OH
EH=int(round(W/ar)); oy=(EH-H)/2           # extended canvas tall enough to show the full photo in 9:16
s=max(W/W,EH/H); bg=img.resize((int(W*s)+2,int(H*s)+2),Image.LANCZOS)
bx=(bg.width-W)//2; bg=bg.crop((bx,0,bx+W,EH)).filter(ImageFilter.GaussianBlur(45))
bg=ImageEnhance.Brightness(bg).enhance(0.55)
mask=Image.new('L',(W,H),255).crop((0,0,W,H)); m=Image.new('L',(W,H),0); m.paste(255,(0,14,W,H-14)); m=m.filter(ImageFilter.GaussianBlur(10))
E=bg.copy(); E.paste(img,(0,int(oy)),m)
#            t     cx    cy    window height (photo px)
K=np.array([[0.0,  300,  470,  770],
            [3.4,  925,  470,  770],
            [6.6,  611,  H/2,  EH ],
            [11.0, 640,  352,  500],
            [14.8, 641,  354,  470]])
dur=14.8; N=int(round(dur*fps)); t=np.arange(N)/fps
f=[PchipInterpolator(K[:,0],K[:,i]) for i in (1,2)]+[PchipInterpolator(K[:,0],np.log(K[:,3]))]
enc=subprocess.Popen([FF,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}','-r',str(fps),'-i','-','-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p',out],stdin=subprocess.PIPE)
for tt in t:
    cx,cy,h=float(f[0](tt)),float(f[1](tt)),float(np.exp(f[2](tt))); w=h*ar
    cy+=oy; cx=min(max(cx,w/2),W-w/2) if w<=W else W/2; cy=min(max(cy,h/2),EH-h/2)
    fr=E.resize((OW,OH),Image.LANCZOS,box=(cx-w/2,cy-h/2,cx+w/2,cy+h/2))
    if h<700: fr=fr.filter(ImageFilter.UnsharpMask(radius=1.4,percent=35,threshold=2))
    enc.stdin.write(fr.tobytes())
enc.stdin.close(); enc.wait(); print('frames',N)
