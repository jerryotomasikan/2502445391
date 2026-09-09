from pathlib import Path
import re, html, json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, CondPageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
import pypdfium2 as pdfium
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent;R=OUT/'pdf_render';R.mkdir(parents=True,exist_ok=True)
for name,file in [('PaperTimes','times.ttf'),('PaperTimesB','timesbd.ttf'),('PaperTimesI','timesi.ttf'),('PaperTimesBI','timesbi.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path(r'C:\Windows\Fonts')/file)))
pdfmetrics.registerFontFamily('PaperTimes',normal='PaperTimes',bold='PaperTimesB',italic='PaperTimesI',boldItalic='PaperTimesBI')
pdfmetrics.registerFont(TTFont('PaperMath',r'C:\Windows\Fonts\cambria.ttc',subfontIndex=1))
styles={
 'body':ParagraphStyle('body',fontName='PaperTimes',fontSize=11,leading=13.2,spaceAfter=7,alignment=TA_JUSTIFY,allowWidows=0,allowOrphans=0),
 'title':ParagraphStyle('title',fontName='PaperTimesB',fontSize=18,leading=21,spaceAfter=12,alignment=TA_CENTER),
 'author':ParagraphStyle('author',fontName='PaperTimes',fontSize=9.5,leading=11.5,spaceAfter=5,alignment=TA_CENTER),
 'h1':ParagraphStyle('h1',fontName='PaperTimesB',fontSize=12,leading=14,spaceBefore=12,spaceAfter=7,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='PaperTimesB',fontSize=11,leading=13,spaceBefore=8,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='PaperTimes',fontSize=9.2,leading=11,spaceAfter=7),
 'tabletitle':ParagraphStyle('tabletitle',fontName='PaperTimesB',fontSize=9.2,leading=11,spaceBefore=7,spaceAfter=5,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='PaperTimes',fontSize=9,leading=10.5,alignment=TA_CENTER),
 'cellleft':ParagraphStyle('cellleft',fontName='PaperTimes',fontSize=9,leading=10.5,alignment=TA_LEFT),
 'ref':ParagraphStyle('ref',fontName='PaperTimes',fontSize=9.2,leading=11,spaceAfter=5,wordWrap='LTR',allowWidows=0,allowOrphans=0),
 'eq':ParagraphStyle('eq',fontName='PaperMath',fontSize=11,leading=15,spaceBefore=5,spaceAfter=9,alignment=TA_CENTER)
}
def fmt(s):
    s=html.escape(s);s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'(https?://[^\s]+)',r'<link href="\1" color="black">\1</link>',s)
    return s
W=A4[0]-2*57.6
story=[];lines=(OUT/'Paper_Revised.md').read_text(encoding='utf-8').splitlines();i=0;refs=False;intro=True
eq=['y<sub>i</sub> = I(q<sub>i</sub> ≥ 6). &nbsp;&nbsp;&nbsp; (1)',
    'A<sub>1</sub> = (1/n) ∑<sub>i=1</sub><super>n</super> I(|q<sub>i</sub> − q̂<sub>i</sub>| ≤ 1). &nbsp;&nbsp;&nbsp; (2)',
    'c* = arg max<sub>c∈Q</sub> [ (1/|D|) ∑<sub>i∈D</sub> I(|q<sub>i</sub> − c| ≤ 1) ]. &nbsp;&nbsp;&nbsp; (3)']
while i<len(lines):
    s=lines[i].strip();i+=1
    if not s:continue
    if s.startswith('# '):story.append(Paragraph(fmt(s[2:]),styles['title']));continue
    if s.startswith('## '):
        refs=s[3:]=='References';intro=False;story.append(Paragraph(fmt(s[3:]),styles['h1']));continue
    if s.startswith('### '):story.append(Paragraph(fmt(s[4:]),styles['h2']));continue
    if s.startswith('|'):
        rows=[s]
        while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
        cells=[[x.strip() for x in r.strip('|').split('|')] for r in rows];cells=[cells[0]]+cells[2:];n=len(cells[0])
        widths={3:[.235,.3825,.3825],5:[.24,.19,.19,.19,.19],6:[.22,.17,.10,.18,.10,.23],7:[.235,.145,.04,.12,.14,.17,.15]}[n]
        # Table S5 also has seven columns but needs room for the margin headers,
        # which the error-accounting widths above compress to a single character.
        if 'Required b−c' in cells[0]:widths=[.215,.115,.135,.135,.10,.10,.10]
        # Normalize fractions so no cell can extend beyond the text block.
        total=sum(widths);widths=[W*x/total for x in widths]
        data=[]
        for ri,row in enumerate(cells):
            data.append([Paragraph(('<b>'+fmt(x)+'</b>') if ri==0 else fmt(x),styles['cellleft'] if ci==0 or (n in [6,7] and ci==1) else styles['cell']) for ci,x in enumerate(row)])
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8E8E8')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,8));continue
    if s.startswith('!['):
        source=OUT/re.search(r'\]\((.*?)\)',s)[1]
        with PILImage.open(source) as im:iw,ih=im.size
        visual=Image(str(source),width=W,height=W*ih/iw)
        j=i
        while j<len(lines) and not lines[j].strip():j+=1
        if j<len(lines) and lines[j].startswith('Fig. '):
            caption=Paragraph(fmt(lines[j]),styles['caption']);story.append(KeepTogether([Spacer(1,5),visual,Spacer(1,5),caption]));i=j+1
        else:story.append(visual)
        continue
    if s.startswith('TABLE '):story.append(Paragraph(fmt(s),styles['tabletitle']));continue
    if re.search(r'\([123]\)$',s):
        idx=int(re.search(r'\(([123])\)$',s)[1])-1;story.append(Paragraph(eq[idx],styles['eq']));continue
    if s.startswith('**Abstract'):intro=False
    style=styles['ref'] if refs else (styles['author'] if intro else styles['body'])
    para=Paragraph(fmt(s),style)
    following=next((z.strip() for z in lines[i:] if z.strip()),'')
    if re.search(r'\([123]\)$',following):para.keepWithNext=True
    story.append(KeepTogether([para]) if refs else para)

def page(canvas,doc):
    canvas.saveState();canvas.setFont('PaperTimes',9);canvas.drawCentredString(A4[0]/2,29,str(doc.page));canvas.restoreState()
pdf=OUT/'Paper_Revised.pdf'
# Take the document title from the manuscript itself so PDF properties cannot
# drift away from the heading the way the previous hard-coded title did.
doctitle=next(z.strip()[2:] for z in lines if z.startswith('# '))
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=57.6,leftMargin=57.6,topMargin=52,bottomMargin=48,title=doctitle,author='Heriyanto and Suharjito')
doc.build(story,onFirstPage=page,onLaterPages=page)
pd=pdfium.PdfDocument(str(pdf));page_info=[]
for i in range(len(pd)):
    pg=pd[i];image=pg.render(scale=1.6).to_pil();image.save(R/f'page-{i+1}.png');page_info.append({'page':i+1,'size':image.size})
reader=PdfReader(pdf);(R/'extracted_text.txt').write_text('\n\n'.join(p.extract_text() for p in reader.pages),encoding='utf-8')
(R/'pages.json').write_text(json.dumps(page_info,indent=2))
print('PDF pages',len(pd))
