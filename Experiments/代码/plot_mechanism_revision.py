"""Draw the record-identification mechanism and economic information audit."""
from pathlib import Path
import os,sys,json,math,itertools,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Polygon
import run_retirement_event_optimization as ro
rpb=ro.rpb
ROOT=Path(__file__).resolve().parents[2]

def frontier(records,lower):
    ell=rpb.CONFIG['lead_time'];events=[]
    for i,r in enumerate(records):
        record=math.inf
        for a,t in zip(r['a'],r['cycles']):
            if a<record:
                record=a
                if a>lower:events.append((float(a),i,float(t)))
    c,u,_=ro.observed_replay(records,lower);num=c.sum();den=u.sum();rows=[(lower,100*num/den)]
    for a,g in itertools.groupby(sorted(events),key=lambda x:x[0]):
        for _,i,t in g:
            z=int(records[i]['failed'] and t+ell>=records[i]['age']);nc=1+9*z;nu=min(t+ell,records[i]['age'])
            num+=nc-c[i];den+=nu-u[i];c[i]=nc;u[i]=nu
        rows.append((a,100*num/den))
    return pd.DataFrame(rows,columns=['margin','event_cost'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'Manuscript'/'figures_rerun');args=ap.parse_args();F=args.output;F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.3,'xtick.labelsize':7.8,'ytick.labelsize':7.8,'legend.fontsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75,'lines.linewidth':1.35})
    BLUE='#286b98';ORANGE='#c97c35';GREEN='#20836d';GREY='#72808a';LIGHT='#edf3f7';BLACK='#24333f'
    pub=ROOT/'Experiments'/'公开基准'
    forecasts=pd.read_csv(pub/'results'/'cmapss_primary_forecasts.csv',float_precision='round_trip')
    f=forecasts[(forecasts.dataset=='FD002')&(forecasts['head']=='Linex')&(forecasts.phase=='cal')]
    seqs=[dict(unit_id=int(uid),T=float(g.max_cycle.iloc[0]),cycles=g.cycle.to_numpy(float),point=g.point.to_numpy(),scale=g.scale.to_numpy()) for uid,g in f.groupby('unit_id')]
    audit=pd.read_csv(pub/'results_retirement_event'/'retirement_event_identification_audit.csv',float_precision='round_trip')
    case=audit[(audit.dataset=='FD002')&(audit.seed==20261023)&(audit['head']=='Linex')&(audit.scenario=='Tune_Q85')].iloc[0]
    h=float(case.historical_margin);L=float(case.risk_floor);star=float(case.selected_margin)
    records=[ro.observe(s,h) for s in seqs]
    choices=[]
    for s,r in zip(seqs,records):
        if r['failed']:continue
        a=(s['point']-15)/s['scale'];nh=np.flatnonzero(a<=h);ns=np.flatnonzero(a<=star)
        th=float(s['cycles'][nh[0]]);ts=float(s['cycles'][ns[0]])
        if 12<=th-ts<=70:choices.append((abs((th-ts)-35),s,r,th,ts))
    _,unit,rec,th,ts=min(choices,key=lambda x:x[0]);T=unit['T'];ell=15.
    fr=frontier(records,h)
    oldcw=rpb.CONFIG['waste_cost'];rpb.CONFIG['waste_cost']=0.
    try:_,_,oracle=rpb.exact_margin_frontier(seqs,h)
    finally:rpb.CONFIG['waste_cost']=oldcw
    oracle=pd.DataFrame(oracle).rename(columns={'j_life':'event_cost'})
    assert len(fr)==len(oracle) and np.max(abs(fr.event_cost-oracle.event_cost))<1e-10
    fr.to_csv(F/'mechanism_observed_frontier.csv',index=False)
    pd.DataFrame({'cycle':unit['cycles'],'score':(unit['point']-15)/unit['scale'],'observed':unit['cycles']<=th}).to_csv(F/'mechanism_example_trajectory.csv',index=False)

    fig=plt.figure(figsize=(7.2,4.65))
    top=fig.add_axes([.025,.665,.95,.29]);top.axis('off');top.set_xlim(-.012,1.012);top.set_ylim(0,1)
    top.text(0,1.04,'(a)  From sensor prefixes to a calibrated replacement policy',fontsize=9.2,fontweight='bold',color=BLACK)
    blocks=[(0.,.215,'Frozen\nforecast map',LIGHT),(.265,.22,'Asset-level\ninformation','#fbf4eb'),(.535,.20,'Identify policy\noutcomes','#edf4f1'),(.785,.215,'Calibrate and\noptimize','#edf3f7')]
    for x,w,title,col in blocks:
        top.add_patch(FancyBboxPatch((x,.035),w,.86,boxstyle='round,pad=.006,rounding_size=.015',fc=col,ec='#94a3ad',lw=.8))
        top.text(x+w/2,.79,title,ha='center',va='center',fontsize=8.5,fontweight='bold',color=BLACK,linespacing=1.05)
    # Prefix channels and a fitted map: these are the implemented features/head.
    for j,col in enumerate([BLUE,GREY,GREEN]):
        xx=np.linspace(.018,.095,35);yy=.64-j*.095+.022*np.sin(np.linspace(0,9,35)+j)
        top.plot(xx,yy,color=col,lw=.85)
    top.annotate('',xy=(.119,.55),xytext=(.101,.55),arrowprops=dict(arrowstyle='->',color=GREY,lw=.8))
    top.text(.161,.585,'60 causal\nfeatures',ha='center',va='center',fontsize=7.5)
    top.text(.108,.29,r'$f_\theta(\mathcal{F}_t),\ s_\theta(\mathcal{F}_t)$',ha='center',fontsize=9)
    top.text(.108,.12,'Fit / Tune; then frozen',ha='center',fontsize=7.4,color=GREY)
    # Two observation types encode complete scores or upper bounds.
    top.text(.282,.63,'Complete life',fontsize=7.8,color=BLUE,fontweight='bold')
    top.text(.282,.485,r'$T_u,\ B_u,\ U_u,\ Z_u,\ W_u$',fontsize=9)
    top.plot([.281,.468],[.40,.40],color='#ccd3d8',lw=.7)
    top.text(.282,.315,'Preventive retirement',fontsize=7.8,color=ORANGE,fontweight='bold')
    top.text(.282,.17,r'$a_1,\ldots,a_{\tau},\ U_u^h,\ Z_u^h$',fontsize=9)
    # Nested alarms identify the same candidate outcomes.
    top.text(.635,.63,r'$\lambda\geq h\ \Rightarrow\ \tau_\lambda\leq\tau_h$',ha='center',fontsize=9)
    for y,end,c in [(.45,.699,ORANGE),(.35,.655,GREEN),(.25,.604,BLUE)]:
        top.plot([.558,.706],[y,y],color='#ccd3d8',lw=.8);top.plot([.558,end],[y,y],color=c,lw=2);top.plot(end,y,'o',ms=3,color=c)
    top.text(.635,.10,r'$U_u(\lambda),\ Z_u(\lambda)$',ha='center',fontsize=9)
    top.text(.892,.62,r'$L_R=\max(0,h,q_R)$',ha='center',fontsize=9,color=GREEN)
    top.text(.892,.41,'Sort prefix records\nUpdate cost / service',ha='center',va='center',fontsize=7.8,linespacing=1.35)
    top.text(.892,.17,'Select best margin',ha='center',fontsize=8,color=BLUE)
    for a,b in [(.217,.261),(.487,.53),(.737,.781)]:top.annotate('',xy=(b,.48),xytext=(a,.48),arrowprops=dict(arrowstyle='->',color=GREY,lw=1.1))
    ax=fig.add_axes([.09,.25,.395,.33]);ax2=fig.add_axes([.60,.25,.375,.33])
    ax.set_title('(b) Nested alarms preserve observable outcomes',loc='left',fontweight='bold',pad=13,fontsize=8.8)
    a=(unit['point']-15)/unit['scale'];x=unit['cycles'];mask=x<=th
    ax.plot(x[mask],a[mask],color=BLUE,lw=1.5,label='Observed prefix')
    ax.plot(x[~mask],a[~mask],color='#a8b0b6',ls='--',lw=1,label='Withheld continuation')
    ax.axvspan(th,th+ell+20,fc='#f1f3f5',zorder=-3)
    ax.axhline(h,color=ORANGE,ls='--',lw=1)
    ax.axhline(star,color=GREEN,ls='--',lw=1)
    ax.axvline(th,color=ORANGE,lw=.75,ls=':');ax.axvline(ts,color=GREEN,lw=.75,ls=':')
    ax.plot(ts,a[np.where(x==ts)[0][0]],'o',ms=4,color=GREEN)
    ax.plot(th,a[np.where(x==th)[0][0]],'o',ms=4,color=ORANGE)
    lo=max(1,ts-15);hi=min(T-1,th+ell+20)
    v=a[(x>=lo)&(x<=hi)];ax.set(xlim=(lo,hi),ylim=(min(v)-.4,max(v)+.45),xlabel='Inspection cycle',ylabel='Normalized forecast score')
    ax.text(lo+1,star+.15,'Candidate margin',fontsize=7.8,color=GREEN)
    ax.text(lo+1,h-.4,r'Historical $h$',fontsize=7.6,color=ORANGE)
    ax.legend(loc='upper right',fontsize=7.5,frameon=False)
    ax.annotate('',xy=(ts+ell,-.40),xytext=(ts,-.40),xycoords=('data','axes fraction'),textcoords=('data','axes fraction'),arrowprops=dict(arrowstyle='|-|',lw=.9,color=GREEN),annotation_clip=False)
    ax.annotate('',xy=(th+ell,-.57),xytext=(th,-.57),xycoords=('data','axes fraction'),textcoords=('data','axes fraction'),arrowprops=dict(arrowstyle='|-|',lw=.9,color=ORANGE),annotation_clip=False)
    ax.text(ts+ell/2,-.35,'15 cycles',transform=ax.get_xaxis_transform(),ha='center',fontsize=7.5,color=GREEN)
    ax.text(th+ell/2,-.68,'15 cycles',transform=ax.get_xaxis_transform(),ha='center',fontsize=7.5,color=ORANGE)
    ax2.set_title('(c) Search the identified cost frontier',loc='left',fontweight='bold',pad=13,fontsize=8.8)
    xmax=star*1.55;vf=fr[fr.margin<=xmax]
    ax2.axvspan(h-.5,h,color='#e0e4e7',zorder=-3)
    ax2.axvspan(h,L,color='#f5eee3',zorder=-3)
    ax2.step(vf.margin,vf.event_cost,where='post',color=BLUE,lw=1.4,label='Observed-record frontier')
    ov=oracle[(oracle.margin>=L)&(oracle.margin<=xmax)]
    step=max(1,len(ov)//18);ax2.plot(ov.margin.iloc[::step],ov.event_cost.iloc[::step],ls='none',marker='o',ms=3,mfc='white',mec=ORANGE,mew=.75,label='Complete-data reference')
    ax2.axvline(L,color=GREEN,ls='--',lw=1)
    ax2.plot(star,case.event_objective,marker='*',ms=10,color=GREEN,zorder=5)
    ax2.annotate(r'$\lambda^*=3.933$'+'\n'+r'$\widehat{J}_e=0.5795$',xy=(star,case.event_objective),xytext=(9,15),textcoords='offset points',fontsize=9,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,lw=.7))
    ax2.set(xlabel=r'Safety margin $\lambda$',ylabel='Cal event cost\n/ 100 service cycles',xlim=(h-.5,xmax))
    ax2.legend(loc='upper right',fontsize=7.5,frameon=False)
    ax2.text(L+.05,.98,r'$L_R$',transform=ax2.get_xaxis_transform(),va='top',color=GREEN,fontsize=9)
    ax2.text(.02,-.47,'48 / 52 natural lifetimes censored\nThe full safe optimum is retained',transform=ax2.transAxes,fontsize=8,color=GREY,linespacing=1.5)
    ax2.grid(axis='y',alpha=.15)
    for ext in ['pdf','png']:fig.savefig(F/('fig1_decision_framework.'+ext),bbox_inches='tight',pad_inches=.045,dpi=240)
    plt.close(fig)
    # Decision information, rather than raw quantile discrepancy, is the focus.
    info=pd.read_csv(pub/'results_information_audit'/'information_audit_configurations.csv')
    curve=pd.read_csv(pub/'results_information_audit'/'historical_threshold_curve.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1,1.1]})
    ax=axs[0];ax.set_title('(a) Economic information in 144 configurations',loc='left',fontweight='bold',pad=12,fontsize=8.8)
    for head,col,mark in [('Ridge',BLUE,'o'),('Linex',ORANGE,'^')]:
        d=info[info['head']==head];ax.scatter(d.censor_pct,d.relative_gap_pct,s=24,color=col,marker=mark,alpha=.7,label=head)
    ax.axhline(0,c=GREY,lw=.7);ax.set(xlabel='Preventively censored Cal assets (%)',ylabel='Empirical policy-range penalty (%)',ylim=(-.28,5.6));ax.legend(loc='upper left',frameon=False)
    ax.text(.05,.65,'130 / 144: same full safe optimum\n39 / 53: same at censoring ≥90%',transform=ax.transAxes,fontsize=8,color=BLACK,linespacing=1.45)
    ax.grid(alpha=.13)
    ax=axs[1];ax.set_title('(b) Historical threshold sets the boundary',loc='left',fontweight='bold',pad=12,fontsize=8.8)
    ax.plot(curve.historical_margin,curve.identified_objective,color=BLUE,lw=1.7,label=r'Identified minimum $V_h$')
    ax.axhline(curve.complete_safe_objective.iloc[0],c=ORANGE,ls='--',lw=1.2,label=r'Full safe minimum $V_0$')
    l0=curve.complete_safe_margin.iloc[0];ax.axvline(l0,c=GREEN,ls=':',lw=1)
    ax.annotate(r'$\lambda_0^*=3.933$',xy=(l0,.606),xytext=(1.05,.606),color=GREEN,fontsize=9,ha='left',va='center',arrowprops=dict(arrowstyle='-',color=GREEN,lw=.8))
    ax.fill_between(curve.historical_margin,curve.complete_safe_objective,curve.identified_objective,color=BLUE,alpha=.12)
    ax.set(xlabel=r'Historical margin $h$ (fixed FD002 Linex fleet)',ylabel='Cal event cost / 100 service cycles')
    ax.legend(loc='upper left',fontsize=9,frameon=False);ax.grid(alpha=.13)
    fig.tight_layout(w_pad=1.5)
    for ext in ['pdf','png']:fig.savefig(F/('fig4_censoring_identification.'+ext),bbox_inches='tight',pad_inches=.045,dpi=240)
    plt.close(fig)
    result=dict(dataset='FD002',head='Linex',phase='Cal',history='Tune_Q85',n_cal=52,n_censored=48,example_unit_id=unit['unit_id'],historical_alarm=th,candidate_alarm=ts,h=h,risk_floor=L,selected_margin=star,event_minimum=float(case.event_objective),frontier_candidate_count=len(fr),max_observed_complete_frontier_error=float(np.max(abs(fr.event_cost-oracle.event_cost))))
    (F/'mechanism_figure_data.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import pymupdf
    geometry=[]
    for p in sorted(F.glob('fig*.pdf')):
        with pymupdf.open(p) as doc:
            sizes=[s['size'] for pg in doc for b in pg.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans'] if s['text'].strip()]
            geometry.append(dict(file=p.name,width_mm=doc[0].rect.width/72*25.4,height_mm=doc[0].rect.height/72*25.4,min_font_pt=min(sizes),max_font_pt=max(sizes)))
    (F/'figure_geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
