from pathlib import Path
from collections import Counter, defaultdict
from hashlib import sha256
import csv, json, math
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path('/Users/neeliamedhani/Machine Vision/FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007')
IMG=ROOT/'JPEGImages'; ANN=ROOT/'Annotations'; MAIN=ROOT/'ImageSets/Main'

def cramer(table):
    rows=list(table); cols=sorted({c for r in rows for c in table[r]})
    obs=[[table[r].get(c,0) for c in cols] for r in rows]
    n=sum(map(sum,obs)); rs=list(map(sum,obs)); cs=[sum(row[j] for row in obs) for j in range(len(cols))]
    chi=0
    for i,row in enumerate(obs):
        for j,v in enumerate(row):
            e=rs[i]*cs[j]/n
            if e: chi+=(v-e)**2/e
    d=n*min(len(rows)-1,len(cols)-1)
    return math.sqrt(chi/d) if d else 0

train=set((MAIN/'trainval.txt').read_text().split())
test=set((MAIN/'test.txt').read_text().split())
env={}
with (MAIN/'CategorizationData/FOD_categorization_annotations.csv').open(newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f): env[Path(r['File']).stem]=(r['Weather'],r['Light'])

classes=Counter(); weather=Counter(); light=Counter(); split_counts=Counter(); objects_per_image=Counter()
class_weather=defaultdict(Counter); class_light=defaultdict(Counter); class_split=defaultdict(Counter)
box_areas=[]; box_widths=[]; box_heights=[]; tiny=Counter(); invalid=[]; unreadable=[]; dims=Counter(); modes=Counter()
hashes=defaultdict(list); split_hashes=defaultdict(set); dhashes=defaultdict(list)
files=sorted(IMG.glob('*.jpg'))
for k,p in enumerate(files,1):
    stem=p.stem; split='trainval' if stem in train else 'test' if stem in test else 'unassigned'
    split_counts[split]+=1
    wcode,lcode=env.get(stem,('missing','missing')); weather[wcode]+=1; light[lcode]+=1
    xml=ANN/(stem+'.xml')
    try:
        root=ET.parse(xml).getroot(); size=root.find('size'); W=int(size.findtext('width')); H=int(size.findtext('height'))
        objs=root.findall('object'); objects_per_image[len(objs)]+=1
        for obj in objs:
            cls=obj.findtext('name'); b=obj.find('bndbox')
            x1,y1,x2,y2=map(float,[b.findtext('xmin'),b.findtext('ymin'),b.findtext('xmax'),b.findtext('ymax')])
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H): invalid.append((stem,cls,x1,y1,x2,y2,W,H))
            bw=x2-x1; bh=y2-y1; area=bw*bh/(W*H)
            classes[cls]+=1; class_weather[cls][wcode]+=1; class_light[cls][lcode]+=1; class_split[cls][split]+=1
            box_areas.append(area); box_widths.append(bw/W); box_heights.append(bh/H)
            if area < .01: tiny[cls]+=1
    except Exception as e: invalid.append((stem,'XML_ERROR',str(e)))
    try:
        data=p.read_bytes(); h=sha256(data).hexdigest(); hashes[h].append(stem); split_hashes[split].add(h)
        with Image.open(p) as im:
            im.load(); dims[im.size]+=1; modes[im.mode]+=1
            g=im.convert('L').resize((9,8)); px=list(g.getdata()); bits=''.join('1' if px[y*9+x]>px[y*9+x+1] else '0' for y in range(8) for x in range(8))
            dhashes[f'{int(bits,2):016x}'].append(stem)
    except Exception as e: unreadable.append((stem,str(e)))
    if k%5000==0: print('processed',k,flush=True)

def pct(v):
    s=sorted(v)
    def q(x): return s[round((len(s)-1)*x)] if s else None
    return {'min':q(0),'p10':q(.1),'median':q(.5),'p90':q(.9),'max':q(1)}

dups=[v for v in hashes.values() if len(v)>1]
cross=[v for v in dups if any(x in train for x in v) and any(x in test for x in v)]
dhash_groups=[v for v in dhashes.values() if len(v)>1]
result={
 'images':len(files),'annotations':len(list(ANN.glob('*.xml'))),'objects':sum(classes.values()),'classes':dict(classes),
 'weather':dict(weather),'light':dict(light),'splits':dict(split_counts),'split_overlap_ids':len(train&test),
 'objects_per_image':dict(objects_per_image),'invalid_boxes_or_xml':len(invalid),'unreadable':len(unreadable),
 'dimensions':{str(k):v for k,v in dims.items()},'modes':dict(modes),
 'box_area_fraction':pct(box_areas),'box_width_fraction':pct(box_widths),'box_height_fraction':pct(box_heights),
 'tiny_objects_under_1pct':sum(tiny.values()),'tiny_by_class':dict(tiny),
 'exact_duplicate_groups':len(dups),'exact_duplicate_images':sum(len(v) for v in dups),
 'cross_split_exact_duplicate_groups':len(cross),'cross_split_examples':cross[:20],
 'dhash_collision_groups':len(dhash_groups),'dhash_collision_images':sum(len(v) for v in dhash_groups),
 'class_weather_cramers_v':cramer(class_weather),'class_light_cramers_v':cramer(class_light),
 'class_split_cramers_v':cramer(class_split),
 'class_weather':{k:dict(v) for k,v in class_weather.items()},'class_light':{k:dict(v) for k,v in class_light.items()},
 'class_split':{k:dict(v) for k,v in class_split.items()},
 'invalid_examples':invalid[:20],'unreadable_examples':unreadable[:20]
}
out=Path('/Users/neeliamedhani/Machine Vision/tmp/fod_audit_results.json')
out.write_text(json.dumps(result,indent=2))
print(out)
