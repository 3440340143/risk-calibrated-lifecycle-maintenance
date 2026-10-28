from pathlib import Path
import shutil,sys,json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import argparse
OUT=Path(__file__).resolve().parents[2]
BASE=OUT
ap=argparse.ArgumentParser(description='Rebuild the five main-paper vector figures from archived records.')
ap.add_argument('--output',type=Path,default=OUT/'Manuscript'/'figures_rerun')
args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
  'axes.titlesize':9.5,'xtick.labelsize':8.3,'ytick.labelsize':8.3,'legend.fontsize':8,
  'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
  'axes.linewidth':.7,'lines.linewidth':1.3,'savefig.dpi':400})
BLUE='#2471a3'; ORANGE='#c86b22'; GREEN='#218c74'; GREY='#5f6a6a';LIGHT='#eef3f6'
F=args.output
def save(fig,name):
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.025)
    fig.savefig(F/(name+'.png'),bbox_inches='tight',pad_inches=.025,dpi=220)
    plt.close(fig)
def panel(ax,l,title):ax.set_title(f'({l}) {title}',loc='left',fontweight='bold',pad=9)

# Figure 1: the distinct observation channels and common decision engine.
fig,ax=plt.subplots(figsize=(7.05,2.60));ax.set_position([.02,.02,.96,.96]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009',fc=color,ec='#7e8b93',lw=.8))
    ax.text(x+w/2,y+h-.045,title,ha='center',va='top',fontweight='bold',fontsize=9)
    ax.text(x+w/2,y+h-.145,body,ha='center',va='top',fontsize=8.5,linespacing=1.30)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=GREY,lw=1.1))
box(.015,.55,.28,.42,'Complete histories','Freeze model on Fit + Tune\nCal: breakdown-risk floor\nTune + Cal: lifecycle cost',LIGHT)
box(.015,.035,.28,.42,'Retirement records','Preventive: prefix and age\nBreakdown: full failure record\nSame historical score map', '#f2f6f0')
box(.385,.20,.275,.61,'Trajectory decision layer',r'$Z_u(\lambda)=\mathbf{1}\{B_u>\lambda\}$'+'\n\nCalibrate the risk floor\nSweep prefix-record thresholds\n$O(M+K\\log K)$', '#e7f0f7')
box(.755,.20,.23,.61,'Select a policy','Complete data:\nlifecycle-cost optimum\n\nRetirement records:\nevent-cost optimum\nfor $\\lambda\\geq\\lambda_{hist}$', '#edf4f1')
arrow(.30,.75,.38,.62);arrow(.30,.24,.38,.37);arrow(.665,.50,.75,.50)
save(fig,'fig1_decision_framework')

P=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_full_pool'/'cmapss_upgraded_primary_pooled.csv').set_index('method')
B=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_full_pool'/'cmapss_upgraded_paired_bootstrap.csv')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.75),gridspec_kw={'width_ratios':[1,1.18]})
ax=axs[0];panel(ax,'a','Primary engine replay')
labels={'Linex_FP_CCPO':'Linex full pool','Linex_ERM_EqualBudget':'Linex equal-budget ERM','Cycle_CP':'Cycle CP','Ridge_FP_CCPO':'Ridge full pool'}
offsets={'Linex_FP_CCPO':(-5,-22),'Linex_ERM_EqualBudget':(-14,12),'Cycle_CP':(7,8),'Ridge_FP_CCPO':(-8,12)}
for name,label in labels.items():
    r=P.loc[name];color=BLUE if 'Linex_FP' in name else ORANGE if 'EqualBudget' in name else GREEN if name=='Cycle_CP' else GREY
    ax.scatter(r.fail_pct,r.j_life,s=42,c=color,marker='s' if 'FP' in name else 'o',zorder=3)
    ax.annotate(label,(r.fail_pct,r.j_life),xytext=offsets[name],textcoords='offset points',fontsize=7.7,ha='left' if name=='Cycle_CP' else 'center')
ax.axvline(5,c=GREY,ls='--',lw=.9);ax.text(5.08,1.154,'5% target',fontsize=7.5,va='top',color=GREY)
ax.set(xlabel='Observed breakdowns (%)',ylabel='Lifecycle cost / 100 service cycles',xlim=(0,8.1),ylim=(1.055,1.158))
ax.grid(alpha=.15)
ax=axs[1];panel(ax,'b','Paired lifecycle-cost differences')
comps=[('Linex_FP_CCPO','Linex_ERM_Tune','Linex vs Tune ERM'),('Linex_FP_CCPO','Linex_ERM_EqualBudget','Linex vs equal-budget ERM'),('Ridge_FP_CCPO','Ridge_Floor','Ridge vs split floor'),('Ridge_FP_CCPO','Ridge_ERM_EqualBudget','Ridge vs equal-budget ERM')]
for i,(t,c,l) in enumerate(comps):
    r=B[(B.treatment==t)&(B.comparator==c)].iloc[0]
    ax.errorbar(r.delta_j_life,i,xerr=[[r.delta_j_life-r.ci95_j_low],[r.ci95_j_high-r.delta_j_life]],fmt='o',ms=4,color=BLUE if 'Linex' in t else GREY,capsize=3)
ax.axvline(0,c=GREY,lw=.8,ls='--');ax.set_yticks(range(4),[c[2] for c in comps]);ax.invert_yaxis();ax.set_xlabel('Cost difference with 95% bootstrap interval')
ax.tick_params(axis='y',labelsize=7.8);fig.tight_layout(w_pad=1.2);save(fig,'fig2_cmapss_cost_risk')

# Figure 3: objective optimization is on development labels, not test outcomes.
sys.path.insert(0,str(BASE/'Experiments'/'代码'));import run_public_benchmark as rpb
fore=pd.read_csv(BASE/'Experiments'/'公开基准'/'results'/'cmapss_primary_forecasts.csv')
sub=fore[(fore.dataset=='FD003')&(fore['head']=='Linex')]
def sequences(phase):
    x=sub[sub.phase==phase]
    return [dict(unit_id=int(uid),T=float(g.max_cycle.iloc[0]),cycles=g.cycle.to_numpy(float),point=g.point.to_numpy(),scale=g.scale.to_numpy()) for uid,g in x.groupby('unit_id')]
seqs=sequences('tune')+sequences('cal');q,_,_=rpb.calibration_threshold(sequences('cal'))
erm,_,front=rpb.exact_margin_frontier(seqs);cc,_,_=rpb.exact_margin_frontier(seqs,q)
front=pd.DataFrame(front);front.to_csv(F/'illustrative_development_frontier.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(7.05,2.5));ax=axs[0];panel(ax,'a','Development frontier: FD003')
limit=max(erm,cc,q)*1.8; v=front[front.margin<=limit]
ax.step(v.margin,v.j_life,where='post',color=BLUE)
ax.axvline(q,ls='--',color=ORANGE,lw=1,label='Risk floor')
ax.scatter([cc],[rpb.summary(rpb.at_margin(seqs,cc))['j_life']],c=GREEN,s=40,zorder=3,label='Constrained optimum')
ax.set(xlabel='Safety margin',ylabel='Empirical lifecycle cost',xlim=(0,limit));ax.legend(loc='upper right',fontsize=7.5)
six=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_full_pool'/'cmapss_upgraded_seed_pooled.csv')
ax=axs[1];panel(ax,'b','Sensitivity across six splits')
seed_order=[20261023,11,22,33,44,55]
for method,label,col,mark in [('Linex_ERM_EqualBudget','Equal-budget ERM',GREY,'o'),('Linex_FP_Floor','Full-pool floor',ORANGE,'^'),('Linex_FP_CCPO','Exact constrained',BLUE,'s')]:
    g=six[six.method==method].set_index('seed').loc[seed_order]
    ax.plot(range(6),g.j_life,marker=mark,ms=4,color=col,label=label,lw=1)
ax.set_xticks(range(6),['Primary','11','22','33','44','55']);ax.set_xlabel('Split seed');ax.set_ylabel('Test lifecycle cost');ax.legend(fontsize=7.3)
fig.tight_layout(w_pad=1.8);save(fig,'fig3_frontier_split_sensitivity')

# Figure 4: historical-threshold imputation is a strong matched comparator.
rc=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_full_pool'/'cmapss_right_censored_verification.csv')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.65));ax=axs[0];panel(ax,'a','Calibration-score recovery')
for head,col,mark in [('Ridge',BLUE,'o'),('Linex',ORANGE,'^')]:
    g=rc[rc['head']==head];ax.scatter(g.censoring_pct,g.exact_q_diff,s=18,alpha=.65,c=col,marker=mark,label=head)
ax.set(xlabel='Preventive censoring in calibration (%)',ylabel=r'$|q_{retirement}-q_{complete}|$');ax.legend(loc='upper left',fontsize=8);ax.grid(alpha=.15)
ax=axs[1];panel(ax,'b','Identified threshold region')
x=np.linspace(-2,3,200);h=0.
ax.plot(x,np.maximum(h,x),color=BLUE,lw=2,label='Complete-data envelope')
ax.plot(x[::10],np.maximum(h,x[::10]),ls='none',marker='o',mfc='white',mec=ORANGE,ms=4,label='Retirement / threshold-bound')
ax.axvline(h,c=GREY,ls='--',lw=.8);ax.axhline(h,c=GREY,ls=':',lw=.8)
ax.set(xlabel='Complete quantile - historical margin',ylabel='Envelope margin - historical margin')
ax.text(-1.8,1.5,'Lower tail:\nnot identified',fontsize=8,color=GREY);ax.text(.65,.28,'Upper tail:\nexact recovery',fontsize=8,color=BLUE)
ax.legend(fontsize=7.1,loc='upper left');fig.tight_layout(w_pad=1.5);save(fig,'fig4_censoring_identification')

# Figure 5: the new retirement-only event-cost objective has its own units.
events=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_retirement_event'/'retirement_event_primary.csv')
events=events[events.scenario=='Tune_ERM']
paired=pd.read_csv(BASE/'Experiments'/'公开基准'/'results_retirement_event'/'retirement_event_paired_bootstrap.csv')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.55));ax=axs[0];panel(ax,'a','Retirement-only event cost')
names=['Historical','Retirement_floor','Retirement_event_opt'];labs=['Historical','Risk floor','Exact sweep']
for j,(head,col) in enumerate([('Ridge',BLUE),('Linex',ORANGE)]):
    g=events[events['head']==head].set_index('method').loc[names]
    xx=np.arange(3)+(j-.5)*.32;ax.bar(xx,g.event_cost,width=.30,color=col,alpha=.87,label=head)
ax.set_xticks(range(3),labs,fontsize=7.5);ax.set_ylabel('Event cost / 100 service cycles');ax.set_ylim(0,.93);ax.legend(ncol=2,loc='upper right',fontsize=7.5)
ax=axs[1];panel(ax,'b','Increment beyond the risk floor')
for i,head in enumerate(['Ridge','Linex']):
    r=paired[(paired['head']==head)&(paired.comparator=='Retirement_floor')].iloc[0]
    ax.errorbar(r.delta_event_cost,i,xerr=[[r.delta_event_cost-r.ci95_low],[r.ci95_high-r.delta_event_cost]],fmt='o',ms=5,color=BLUE if i==0 else ORANGE,capsize=3)
ax.axvline(0,c=GREY,lw=.8,ls='--');ax.set_yticks([0,1],['Ridge','Linex']);ax.set_ylim(-.5,1.5);ax.invert_yaxis();ax.set_xlabel('Event-cost difference with 95% interval');ax.grid(axis='x',alpha=.15)
fig.tight_layout(w_pad=1.5);save(fig,'fig5_retirement_only_cost')

audit=[]
import pymupdf
for p in F.glob('*.pdf'):
    d=pymupdf.open(p);sz=[s['size'] for pg in d for b in pg.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
    audit.append(dict(file=p.name,width_mm=d[0].rect.width/72*25.4,height_mm=d[0].rect.height/72*25.4,min_font_pt=min(sz),max_font_pt=max(sz)))
(F/'figure_geometry.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))

