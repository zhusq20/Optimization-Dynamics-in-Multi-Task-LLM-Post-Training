from pathlib import Path
from copy import deepcopy
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, ArrayObject, TextStringObject, ByteStringObject, NameObject, RectangleObject

ROOT=Path.cwd()
BACKUP=ROOT/'tmp/pdfs/figure_sources'
BACKUP.mkdir(exist_ok=True)

def compose(a,b):
    A,B,C,D,E,F=a; g,h,i,j,k,l=b
    return (A*g+C*h,B*g+D*h,A*i+C*j,B*i+D*j,A*k+C*l+E,B*k+D*l+F)
def point(m,x,y):
    a,b,c,d,e,f=m; return a*x+c*y+e,b*x+d*y+f

def inverse_point(m,x,y):
    a,b,c,d,e,f=m; det=a*d-b*c
    return (d*(x-e)-c*(y-f))/det,(-b*(x-e)+a*(y-f))/det

def compact(path, lo,hi,newlo,newhi,newheight, replacements):
    src=BACKUP/path.name
    if not src.exists(): src.write_bytes(path.read_bytes())
    reader=PdfReader(src); page=reader.pages[0]
    stream=ContentStream(page.get_contents(),reader)
    def fy(y):
        if lo>50 and y<lo-18:
            if y<=28:return y
            return 28+(y-28)*(newlo-18-28)/(lo-18-28)
        if y<=lo:return y+newlo-lo
        if y>=hi:return y+newhi-hi
        return newlo+(y-lo)*(newhi-newlo)/(hi-lo)
    old=new=(1.,0.,0.,1.,0.,0.);stack=[]; newops=[]
    for i,(raw,op) in enumerate(stream.operations):
        args=deepcopy(raw)
        if op==b'q':stack.append((old,new))
        elif op==b'Q':old,new=stack.pop()
        elif op==b'cm':
            mat=tuple(map(float,args));oldnext=compose(old,mat)
            a,b,c,d,e,f=oldnext
            # Change positions, keeping font and marker transforms unchanged.
            n=(a,b,c,d,e,fy(f))
            # Images contain heatmaps: resize only their data rectangles.
            if i+1<len(stream.operations) and stream.operations[i+1][1]==b'Do' and a!=1 and not b and not c:
                n=(a,0.,0.,fy(f+d)-fy(f),e,fy(f))
            # Center rotated y-axis labels on the shorter plot.
            elif abs(b)==1 and abs(c)==1 and a==d==0:
                n=(a,b,c,d,e,f+(newlo+newhi-lo-hi)/2)
            p0=inverse_point(new,n[4],n[5]);p1=inverse_point(new,n[4]+n[0],n[5]+n[1]);p2=inverse_point(new,n[4]+n[2],n[5]+n[3])
            args=[p1[0]-p0[0],p1[1]-p0[1],p2[0]-p0[0],p2[1]-p0[1],p0[0],p0[1]]
            old,new=oldnext,n
        elif op in [b'm',b'l',b'c',b'v',b'y']:
            vals=[]
            for j in range(0,len(args),2):
                x,y=point(old,float(args[j]),float(args[j+1]));vals.extend(inverse_point(new,x,fy(y)))
            args=vals
        elif op==b're':
            x,y,w,h=map(float,args)
            x0,y0=point(old,x,y);x1,y1=point(old,x+w,y+h)
            a,b=inverse_point(new,x0,fy(y0));c,d=inverse_point(new,x1,fy(y1));args=[a,b,c-a,d-b]
        elif op==b'TJ':
            text=''.join(x for x in args[0] if isinstance(x,str))
            if text in replacements: args=[ArrayObject([ByteStringObject(replacements[text].encode("utf-16-be"))])]
        if op in [b'cm',b'm',b'l',b'c',b'v',b'y',b're']:args=[FloatObject(v) for v in args]
        newops.append((args,op))
    stream.operations=newops
    page[NameObject('/Contents')]=stream
    page.mediabox=RectangleObject([0,0,float(page.mediabox.width),newheight]);page.cropbox=page.mediabox
    writer=PdfWriter();writer.add_page(page)
    with path.open('wb') as f:writer.write(f)
    print(path, 'height',float(reader.pages[0].mediabox.height),'->',newheight)

compact(ROOT/'figures/added_control_figures_20260914/normalization_interventions.pdf',42.804,125.46,38,102,123,{})
compact(ROOT/'figures/added_control_figures_20260914/sampling_history_controls.pdf',42.804,125.46,36,100,123,{'(c) Reset history':'(c) Reset moments'})
compact(ROOT/'figures/nine_panel_20260914/section5_supervision.pdf',61.488,193.248,49,150,174,{
    '(a) Local BF16 activity':'(a) BF16 activity',
    'Local loss; rows: joint student / update':'Loss; rows: student / update',
    'Cosine to full-vocabulary loss':'Cosine to full vocabulary',
    'Grey: I64 range across all banks':'Grey: I64 range over banks',
    'Update (measured seed 42)':'Update (seed 42)',
})
