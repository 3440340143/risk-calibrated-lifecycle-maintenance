"""Build the revised public-engine figures from the locally reproduced CSVs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.3,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'公开基准/results_track_b_rerun')
    p.add_argument('--output',type=Path,default=ROOT.parent/'Track_B/figures')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    prim=pd.read_csv(args.results/'cmapss_upgraded_primary_pooled.csv').set_index('method')
    boot=pd.read_csv(args.results/'cmapss_upgraded_paired_bootstrap.csv')
    fig,ax=plt.subplots(2,1,figsize=(6.8,6.1),layout='constrained')
    spec=[('Linex_FP_CCPO','Linex full pool','#087f8c','*'),('Linex_ERM_EqualBudget','Linex equal-budget ERM','#bd653c','o'),
          ('Ridge_FP_CCPO','Ridge full pool','#164c76','s'),('Ridge_Floor','Ridge split floor','#76569c','D'),
          ('Cycle_CP','Cycle CP','#687277','^')]
    offsets=[(-2.1,-.019),(-2.15,.009),(.15,-.008),(-2.0,.012),(.22,-.012)]
    for (method,label,color,marker),(dx,dy) in zip(spec,offsets):
        row=prim.loc[method]
        ax[0].scatter(row.fail_pct,row.j_life,color=color,marker=marker,s=95 if marker=='*' else 44,zorder=3)
        ax[0].annotate(label,(row.fail_pct,row.j_life),xytext=(row.fail_pct+dx,row.j_life+dy),fontsize=8.2,
                       arrowprops={'arrowstyle':'-','color':color,'lw':.65})
    ax[0].axvline(5,color='#777777',ls='--',lw=1)
    ax[0].text(5.12,1.059,'5% target',rotation=90,fontsize=8,color='#555555')
    ax[0].set(xlabel='Observed breakdown proportion (%)',ylabel='Lifecycle cost per 100 useful cycles',
              xlim=(0,8),ylim=(1.045,1.172),title='(a) Primary replay on 286 engines')
    comp=[('Linex_FP_CCPO','Linex_ERM_Tune','Linex: full pool vs Tune ERM'),
          ('Linex_FP_CCPO','Linex_ERM_EqualBudget','Linex: full pool vs equal-budget ERM'),
          ('Linex_FP_CCPO','Linex_Constrained','Linex: full pool vs split constrained'),
          ('Ridge_FP_CCPO','Ridge_ERM_Tune','Ridge: full pool vs Tune ERM'),
          ('Ridge_FP_CCPO','Ridge_Floor','Ridge: full pool vs split floor'),
          ('Ridge_FP_CCPO','Ridge_ERM_EqualBudget','Ridge: full pool vs equal-budget ERM')]
    for i,(t,c,label) in enumerate(comp):
        row=boot[(boot.treatment==t)&(boot.comparator==c)].iloc[0]
        ax[1].plot([row.ci95_j_low,row.ci95_j_high],[i,i],color='#164c76',lw=1.7)
        ax[1].scatter(row.delta_j_life,i,color='#087f8c',s=28,zorder=3)
    ax[1].set_yticks(range(len(comp)),[c[2] for c in comp],fontsize=8.2)
    ax[1].invert_yaxis();ax[1].axvline(0,color='#777777',ls='--',lw=1)
    ax[1].set(xlabel='Paired lifecycle cost difference with 95% bootstrap interval',title='(b) Full-pool policy comparisons')
    fig.savefig(args.output/'fig6_cmapss_fp_ccpo.pdf',bbox_inches='tight')
    fig.savefig(args.output/'fig6_cmapss_fp_ccpo.png',bbox_inches='tight');plt.close(fig)
    rc=pd.read_csv(args.results/'cmapss_right_censored_verification.csv')
    primary=rc[rc.seed==20261023]
    fig,ax=plt.subplots(2,1,figsize=(6.8,5.4),layout='constrained')
    scenarios=['Historical_Q70_Tune','Historical_Q85_Tune','Historical_ERM_Tune']
    x=np.arange(3)
    for head,color in [('Ridge','#164c76'),('Linex','#087f8c')]:
        for prefix,style,marker,label in [('oracle','--','o','complete-data reference'),('rc_crc','-','s','retirement score')]:
            vals=[]
            for scenario in scenarios:
                g=primary[(primary['head']==head)&(primary.scenario==scenario)]
                vals.append(100*g[f'{prefix}_total_cost'].sum()/g[f'{prefix}_service_cycles'].sum())
            ax[0].plot(x,vals,color=color,ls=style,marker=marker,label=f'{head} {label}',lw=1.5)
    ax[0].set_xticks(x,['Tune Q70','Tune Q85','Tune ERM'])
    ax[0].set(xlabel='Historical policy',ylabel='Lifecycle cost per 100 useful cycles',title='(a) Calibration after preventive censoring')
    ax[0].legend(fontsize=7.8,loc='best',frameon=False)
    for head,color,marker in [('Ridge','#164c76','o'),('Linex','#087f8c','s')]:
        g=rc[rc['head']==head]
        ax[1].scatter(g.censoring_pct,g.envelope_threshold_error,color=color,marker=marker,s=24,alpha=.55,label=head)
    ax[1].axhline(0,color='#777777',lw=.8,zorder=0)
    ax[1].set(xlabel='Preventively censored calibration engines (%)',ylabel='Absolute envelope threshold difference',
              title='(b) Upper-tail identity in all 144 replays',ylim=(-.005,.03))
    ax[1].text(60,.018,'All 144 differences equal zero',fontsize=9)
    ax[1].legend(frameon=False,fontsize=8)
    fig.savefig(args.output/'fig7_right_censored_calibration.pdf',bbox_inches='tight')
    fig.savefig(args.output/'fig7_right_censored_calibration.png',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
