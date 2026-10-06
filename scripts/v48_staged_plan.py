"""Deterministic budgeted design; no likelihood or thermal approximation."""
import itertools, json, random
from v48_staged_common import identity
AXES=('compact','extended','tau_C','tau_E','tau_visc','eta')
def key(x): return json.dumps(x,sort_keys=True)
def levels(spec):
    return {k:sorted(spec['axes'][k]) for k in AXES}
def settings(spec):
    lv=levels(spec)
    return [dict(zip(AXES,v)) for v in itertools.product(*(lv[k] for k in AXES))]
def unique(configs):
    return list({identity(c):c for c in configs}.values())
def balanced_design(spec,n=64,seed=48101,exclude=()):
    """Greedy marginal balance over the explicitly listed finite domain."""
    pool=[s for s in settings(spec) if key(s) not in set(exclude)]
    rng=random.Random(seed);rng.shuffle(pool)
    lv=levels(spec);counts={k:{key(v):0 for v in lv[k]} for k in AXES}
    anchor={k:spec['axes'][k][0] for k in AXES};chosen=[]
    if anchor in pool:chosen.append(anchor);pool.remove(anchor)
    for s in chosen:
        for k in AXES:counts[k][key(s[k])]+=1
    while len(chosen)<n:
        if not pool:raise ValueError('Insufficient unique design points')
        # Counts divided by ideal count makes 3-level and 5-level axes comparable.
        s=min(pool,key=lambda s:sum(counts[k][key(s[k])]*len(lv[k]) for k in AXES))
        pool.remove(s);chosen.append(s)
        for k in AXES:counts[k][key(s[k])]+=1
    return chosen

def config(spec,star,setting):return dict(spec['base'],stars=star,**setting)
def stage1(spec):
    design=balanced_design(spec)
    # One setting across ALL stars before moving on to the next setting.
    return [config(spec,star,s) for s in design for star in spec['axes']['stars']]
def valid_rows(rows):
    return sorted((r for r in rows if r['status']=='completed'),key=lambda r:(r['fit']['chi2_total'],r['model_id']))
def stage2(spec,rows):
    first=balanced_design(spec);explore=balanced_design(spec,n=16,seed=48102,exclude=[key(s) for s in first])
    out=[config(spec,star,s) for s in explore for star in spec['axes']['stars']]
    lv=levels(spec);by={star:[] for star in spec['axes']['stars']}
    for r in valid_rows(rows):
        if r['config']['stars'] in by:by[r['config']['stars']].append(r)
    selected=[]
    for star,rs in by.items():
        # Every STARS curve with a supported result gets local exploration.
        # Keep the best plus a best alternative dust geometry when available.
        seeds=rs[:1]
        if rs:
            geom=lambda r:key([r['config']['compact'],r['config']['extended']])
            alt=next((r for r in rs[1:] if geom(r)!=geom(rs[0])),None)
            if alt is None and len(rs)>1:alt=rs[1]
            if alt:seeds.append(alt)
        for r in seeds:
            c=r['config'];selected.append(r['model_id'])
            for k in AXES:
                idx=lv[k].index(c[k])
                for j in (idx-1,idx+1):
                    if 0<=j<len(lv[k]):out.append(dict(c,**{k:lv[k][j]}))
    return unique(out),selected

def fine_selection(rows):
    rs=valid_rows(rows);out=rs[:8];seen={r['config']['stars'] for r in out}
    # Up to 12 additional distinct source curves, avoiding a single-source shortlist.
    extra=0
    for r in rs:
        if r['config']['stars'] not in seen:
            out.append(r);seen.add(r['config']['stars']);extra+=1
            if extra==12:break
    return unique([r['config'] for r in out])
