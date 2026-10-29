#!/usr/bin/env python3
"""Build Reshoot 4D from rectangles and polygons, following brutus glyphs.py.
No font is used for the wordmarks. Run from any working directory.
"""
from pathlib import Path
import base64
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
L, K = '#C8FF00', '#000000'

def R(x, y, w, h):
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"/>'

def P(*pts):
    return '<polygon points="' + ' '.join(f'{x:g},{y:g}' for x, y in pts) + '"/>'

def glyph(c, w=172, h=124, t=40, th=24):
    m = (h-th)/2
    if c == 'R':
        return R(0,0,t,h)+R(0,0,w,th)+R(0,m,w,th)+R(w-t,0,t,m+th)+P((w*.3,m+th*.5),(w*.3+46,m+th*.5),(w,h),(w-46,h))
    if c == 'S':
        return R(0,0,w,th)+R(0,0,t,m+th)+R(0,m,w,th)+R(w-t,m,t,h-m)+R(0,h-th,w,th)
    if c == 'T':
        return R(0,0,w,th)+R((w-t)/2,0,t,h)
    if c == 'E':
        return R(0,0,t,h)+R(0,0,w,th)+R(0,m,w-16,th)+R(0,h-th,w,th)
    if c == 'H':
        return R(0,0,t,h)+R(w-t,0,t,h)+R(0,m,w,th)
    if c == 'O':
        return R(0,0,t,h)+R(w-t,0,t,h)+R(0,0,w,th)+R(0,h-th,w,th)
    if c == '4':
        return P((w*.42,0),(w*.42+t,0),(t,m),(w,m),(w,m+th),(0,m+th),(0,m))+R(w-t-12,0,t,h)
    if c == 'D':
        # big chamfers on the right only, so D never reads as O (the O stays square on all four corners)
        c = h*.42; ci = c*.62
        outer = [(0,0),(w-c,0),(w,c),(w,h-c),(w-c,h),(0,h)]
        inner = [(t,th),(w-t-ci,th),(w-t,th+ci),(w-t,h-th-ci),(w-t-ci,h-th),(t,h-th)]
        d = lambda pts: 'M'+' L'.join(f'{x:g},{y:g}' for x,y in pts)+' Z'
        return f'<path fill-rule="evenodd" d="{d(outer)} {d(inner)}"/>'
    raise ValueError(c)

def word(text):
    x, out = 0, []
    for c in text:
        if c == ' ':
            x += 52
            continue
        out.append(f'<g transform="translate({x} 0)">{glyph(c)}</g>')
        x += 182
    return ''.join(out), x-10

def smooth(name='sm', r=3.6):
    return f'<filter id="{name}" x="-10%" y="-15%" width="120%" height="130%" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="{r}"/><feColorMatrix type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 36 -16"/></filter>'

def doc(w,h,body,defs='',title='Reshoot 4D'):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}"><title>{title}</title><defs>{defs}</defs><rect width="{w}" height="{h}" fill="{K}"/>{body}</svg>\n'

def build():
    out = ROOT/'assets'
    brand = ROOT/'site/media/brand'
    g, width = word('RESHOOT 4D')
    w,h = width+48,172
    plain = doc(w,h,f'<g transform="translate(24 24)" fill="{L}" filter="url(#sm)">{g}</g>',smooth())
    carved = doc(w,h,f'<rect width="{w}" height="{h}" rx="19" fill="{L}"/><g transform="translate(24 24)" fill="{K}" filter="url(#sm)">{g}</g>',smooth(r=4.2))
    # Two rows keep the R4D mark readable in square avatar and favicon slots.
    compact = f'<g transform="translate(115 24)">{glyph("R")}</g><g transform="translate(24 158)">{glyph("4")}</g><g transform="translate(206 158)">{glyph("D")}</g>'
    mark = doc(402,306,f'<rect width="402" height="306" rx="19" fill="{L}"/><g fill="{K}" filter="url(#sm)">{compact}</g>',smooth(r=4.2), 'R4D')
    files = {'reshoot4d_wordmark.svg':plain,'reshoot4d_carved.svg':carved,'reshoot4d_mark.svg':mark,'favicon.svg':mark}
    font = base64.b64encode((brand/'Unbounded-latin.woff2').read_bytes()).decode()
    fontdef = '<style>@font-face{font-family:Unbounded;src:url(data:font/woff2;base64,'+font+') format("woff2");font-weight:200 900}text{font-family:Unbounded,system-ui,sans-serif}</style>'
    # Keep the original brutus geometry and filter, with a unique filter id.
    brutus = (brand/'04_carved_smooth.svg').read_text().replace('id="sm"','id="brutus-sm"').replace('url(#sm)','url(#brutus-sm)')
    brutus = brutus.replace('<svg ', '<svg x="0" y="0" ',1)
    for name,bw,bh in [('banner.svg',1600,500),('social_card.svg',1280,640)]:
        scale = (bw-160)/w
        y = 110 if bh==500 else 170
        body = f'<g transform="translate(80 {y}) scale({scale})"><g fill="{L}" filter="url(#sm)" transform="translate(24 24)">{g}</g></g>'
        # Two lines on the social card prevent small or compressed supporting copy.
        if bh==500:
            body += f'<text x="100" y="{y+172*scale+56:g}" fill="#AAAAAA" font-size="22">Re-shoot any clip from a new camera angle, inside ComfyUI.</text>'
        else:
            body += '<text x="96" y="370" fill="#AAAAAA" font-size="24"><tspan x="96">Re-shoot any clip from a new camera angle,</tspan><tspan x="96" dy="38">inside ComfyUI.</tspan></text>'
        body += f'<path d="M96 {bh-76}h44" stroke="{L}" stroke-width="3"/><text x="{bw-210}" y="{bh-70}" fill="#AAAAAA" font-size="15">by</text><g transform="translate({bw-178} {bh-138}) scale(.18)">{brutus}</g>'
        files[name] = doc(bw,bh,body,smooth()+fontdef,'Reshoot 4D by brutus')
    for name,svg in files.items():
        ET.fromstring(svg)
        (out/name).write_text(svg)
    for name in ('reshoot4d_wordmark.svg','reshoot4d_carved.svg','reshoot4d_mark.svg','favicon.svg'):
        (brand/name).write_text(files[name])
    print(f'Wrote {len(files)} asset SVGs and 4 site copies')

if __name__ == '__main__':
    build()
