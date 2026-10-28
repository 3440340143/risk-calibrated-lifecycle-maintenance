"""Reader-facing figures: explicit graph semantics and unobstructed labels.

All quantities come from the released numerical records. No model is refitted
and no experimental outcome is changed by this presentation revision.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE=Path(__file__).resolve().parents[2]
PUB=BASE/'Experiments'/'公开基准'
DATA=PUB/'results_censored_lifecycle'
FIG=BASE/'Manuscript'/'figures'
DERIVED=BASE/'Experiments'/'derived'
P=dict(navy='#254360',teal='#147D8D',orange='#BF6B30',gray='#75828B',purple='#80629A',light='#F1F5F7')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                     'axes.titlesize':9.5,'axes.titleweight':'bold','xtick.labelsize':8.5,
                     'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,
                     'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.75,'axes.grid':False,'savefig.dpi':360,'mathtext.fontset':'stix'})
cfg=pd.read_csv(DATA/'lifecycle_identification.csv')
states=pd.read_csv(DATA/'lifecycle_candidate_states.csv')


def select(frame,head='Linex',scenario='Tune_Q85',dataset='FD002',seed=20261023):
    return frame[(frame.dataset==dataset)&(frame.seed==seed)&(frame['head']==head)&(frame.scenario==scenario)]


def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(FIG/(name+'.'+ext),bbox_inches='tight',pad_inches=.045)
    plt.close(fig)


def node(ax,xy,size,title,body,color):
    x,y=xy;w,h=size
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.018',
                               facecolor=P['light'],edgecolor=color,lw=1.0))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',weight='bold',fontsize=9)
    ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=8.6,linespacing=1.25)


def arrow(ax,start,end,label=None,offset=(0,.035)):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,lw=1.05,color=P['gray']))
    if label:
        x=(start[0]+end[0])/2+offset[0];y=(start[1]+end[1])/2+offset[1]
        ax.text(x,y,label,ha='center',va='bottom',fontsize=8,color=P['gray'])


def mechanism():
    case=select(cfg).iloc[0]
    fore=pd.read_csv(PUB/'results'/'cmapss_primary_forecasts.csv',float_precision='round_trip')
    f=fore[(fore.dataset=='FD002')&(fore['head']=='Linex')&(fore.phase=='cal')]
    # The example is fixed by unit ID. Its larger separation between alarm times
    # makes the nesting argument visible at the final printed size.
    uid=247;u=f[f.unit_id==uid].sort_values('cycle')
    t=u.cycle.to_numpy();a=(u.point.to_numpy()-15)/u.scale.to_numpy()
    historical=int(t[np.flatnonzero(a<=case.h)[0]])
    candidate=int(t[np.flatnonzero(a<=case.lam_jl_low)[0]])
    failure=int(u.max_cycle.iloc[0]);observed=t<=historical
    assert candidate<=historical and historical+15<failure
    trace=pd.DataFrame(dict(cycle=t,score=a,observed_prefix=observed.astype(int)))
    trace.to_csv(DERIVED/'lifecycle_mechanism_trace.csv',index=False)
    fig=plt.figure(figsize=(7.25,4.55))
    flow=fig.add_axes([.035,.64,.93,.31]);flow.axis('off');flow.set(xlim=(0,1),ylim=(0,1))
    flow.text(0,1.03,'(a) How maintenance records become a replacement decision',weight='bold',fontsize=9.5)
    node(flow,(.015,.47),(.275,.40),'Historical retirement logs','sensor prefix, age, failure',P['navy'])
    node(flow,(.365,.55),(.265,.32),'Counterfactual replay',r'event costs $C^o$, service $U^o$',P['navy'])
    node(flow,(.725,.55),(.265,.32),'Lifecycle cost family',r'$g(\lambda;s)=A_\lambda+B_\lambda s$',P['teal'])
    node(flow,(.365,.045),(.265,.32),'Risk calibration',r'feasible margins $\lambda\geq L_R$',P['navy'])
    node(flow,(.725,.045),(.265,.32),'Replacement policy','stable / minimax choice',P['teal'])
    arrow(flow,(.297,.71),(.355,.71))
    arrow(flow,(.64,.71),(.715,.71))
    flow.plot([.15,.15,.328],[.46,.205,.205],color=P['gray'],lw=1.05)
    arrow(flow,(.328,.205),(.355,.205))
    arrow(flow,(.64,.205),(.715,.205))
    arrow(flow,(.858,.54),(.858,.375))
    ax=fig.add_axes([.09,.16,.36,.36]);right=fig.add_axes([.61,.16,.36,.36])
    # Legends sit above the data axes, outside the curve area.
    ax.plot(t[observed],a[observed],color=P['navy'],lw=1.5,label='Recorded')
    ax.plot(t[~observed],a[~observed],color=P['gray'],ls='--',lw=1.1,label='Hidden suffix')
    ax.axvspan(historical,failure,color='#E6EBEE',alpha=.7,zorder=0)
    ax.axhline(case.h,color=P['gray'],lw=.8,ls=':')
    ax.axhline(case.lam_jl_low,color=P['teal'],lw=.9,ls='--')
    ax.scatter([candidate,historical],[a[t==candidate][0],a[t==historical][0]],
               c=[P['teal'],P['navy']],s=22,zorder=5)
    ax.set(xlim=(max(1,candidate-52),failure),xlabel='Inspection cycle',ylabel='Prognostic score')
    ax.set_title('(b) The candidate alarm is already recorded',loc='left',pad=28,fontsize=9)
    ax.legend(loc='lower left',bbox_to_anchor=(-.03,1.015),ncol=2,frameon=False,
              borderaxespad=0,columnspacing=.8,handlelength=1.5,fontsize=8.3)
    # The two alarm cycles are explicit colored axis ticks; explanatory text is
    # placed in the caption rather than across the measured curve.
    ticks=[100,125,candidate,historical,175,failure]
    ax.set_xticks(ticks,[str(x) for x in ticks])
    for tick,value in zip(ax.get_xticklabels(),ticks):
        if value in [candidate,historical]:
            tick.set_color(P['teal'] if value==candidate else P['navy']);tick.set_weight('bold')
            tick.set_ha('right' if value==candidate else 'left')
    st=select(states);xx=np.linspace(case.S_T_low,case.S_T_up,200)
    costs=100*(st.event_cost_sum.to_numpy()[:,None]+.045*xx[None,:])/st.service_sum.to_numpy()[:,None]-4.5
    for margin,label,color in [(case.lam_je,'Event rule',P['orange']),
                               (case.lam_jl_low,'Lifecycle rule',P['teal'])]:
        i=int(np.argmin(abs(st.margin.to_numpy()-margin)))
        right.plot(xx/case.n_cal,costs[i],color=color,lw=1.7,label=label)
    right.set(xlabel='Lifetime scenario (cycles / asset)',ylabel='Cal lifecycle cost')
    right.set_title('(c) Price the useful life forgone',loc='left',pad=28,fontsize=9)
    right.legend(loc='lower left',bbox_to_anchor=(-.03,1.015),ncol=2,frameon=False,
                 borderaxespad=0,fontsize=8.0,handlelength=1.5,columnspacing=.8)
    save(fig,'fig1_decision_framework')
    (FIG/'mechanism_figure_data.json').write_text(json.dumps(dict(dataset='FD002',head='Linex',seed=20261023,
        phase='Cal',history='Tune_Q85',example_unit_id=uid,historical_alarm=historical,candidate_alarm=candidate,
        h=float(case.h),candidate_margin=float(case.lam_jl_low),n_cal=int(case.n_cal),n_censored=int(case.n_censored)),indent=2),encoding='utf-8')


def scalar_recovery():
    case=select(cfg,head='Ridge',scenario='Tune_ERM').iloc[0]
    st=select(states,head='Ridge',scenario='Tune_ERM').sort_values('margin')
    # Find exact lower-envelope transitions; draw separate horizontal pieces.
    hull=[]
    for _,row in st.iloc[::-1].iterrows():
        b=4.5/row.service_sum;c=100*row.event_cost_sum/row.service_sum-4.5
        line=(b,c,float(row.margin))
        if hull and abs(hull[-1][0][0]-b)<1e-15:
            if c<=hull[-1][0][1]+1e-14:hull.pop()
            else:continue
        cross=-math.inf
        while hull:
            previous,begin=hull[-1]
            cross=(c-previous[1])/(previous[0]-b)
            if cross<=begin:hull.pop()
            else:break
        hull.append((line,cross if hull else -math.inf))
    intervals=[]
    for i,(line,start) in enumerate(hull):
        end=hull[i+1][1] if i+1<len(hull) else math.inf
        lo=max(case.S_T_low,start);hi=min(case.S_T_up,end)
        if hi>lo:intervals.append((lo/case.n_cal,hi/case.n_cal,line[2]))
    assert len(intervals)==2
    # Confirm that each displayed constant decision actually minimizes the
    # complete candidate frontier inside its displayed scalar interval.
    for lo,hi,margin in intervals:
        scalar=.5*(lo+hi)*case.n_cal
        j=int(np.argmin(abs(st.margin.to_numpy()-margin)))
        costs=100*(st.event_cost_sum.to_numpy()+.045*scalar)/st.service_sum.to_numpy()-4.5
        assert abs(costs[j]-costs.min())<1e-10
    fig,ax=plt.subplots(1,3,figsize=(7.25,2.6),gridspec_kw={'width_ratios':[1.05,1.08,1.2]})
    for j,(lo,hi,margin) in enumerate(intervals):
        ax[0].hlines(margin,lo,hi,color=P['teal'],lw=2)
        ax[0].plot(lo,margin,'o',color=P['teal'],ms=4)
        ax[0].plot(hi,margin,'o',mfc='white' if j==0 else P['teal'],mec=P['teal'],ms=4)
    switch=intervals[1][0]
    ax[0].annotate(f'Switch at {switch:.1f}\ncycles / asset',
                    xy=(switch,intervals[1][2]),xytext=(.05,.55),textcoords='axes fraction',
                    fontsize=8.3,color=P['navy'],ha='left',va='center',
                    arrowprops=dict(arrowstyle='->',color=P['gray'],lw=.8))
    ax[0].set(ylim=(intervals[1][2]-.035,intervals[0][2]+.035),
              xlabel='Lifetime scenario (cycles / asset)',ylabel='Selected safety margin')
    ax[0].set_yticks([x[2] for x in intervals],[f'{x[2]:.3f}' for x in intervals])
    ax[0].set_title('(a) Two stable decisions',loc='left',pad=13)
    names=['Event cost','Observable life','Interval minimax','Mean scalar']
    counts=[int(cfg.je_matches_ray.sum()),int(cfg.low_matches_ray.sum()),int(cfg.mm_matches_ray.sum()),int(cfg.mean_matches_ray.sum())]
    ax[1].barh(range(4),counts,color=[P['gray'],P['teal'],P['purple'],P['navy']],height=.54)
    ax[1].set_yticks(range(4),names);ax[1].invert_yaxis()
    for j,n in enumerate(counts):ax[1].text(n+2,j,str(n),va='center',fontsize=8.8,weight='bold')
    ax[1].set(xlim=(0,166),xlabel='Same-ray optima / 144')
    ax[1].set_title('(b) Optimum recovery',loc='left',pad=13)
    values=[]
    for method in ['cal_jl_je','cal_jl_low']:
        values.append([100*(cfg[cfg.dataset==d][method]/cfg[cfg.dataset==d].oracle_ray_jl-1).mean()
                       for d in ['FD001','FD002','FD003','FD004']])
    values=np.maximum(0,np.asarray(values))
    im=ax[2].imshow(values,cmap='Blues',vmin=0,vmax=max(3,values.max()),aspect='auto')
    ax[2].set_xticks(range(4),['FD001','FD002','FD003','FD004'],rotation=45,ha='right')
    ax[2].set_yticks([0,1],['Event cost','Observable life'])
    for i in range(2):
        for j in range(4):
            number=f'{values[i,j]:.3f}' if values[i,j] and values[i,j]<.01 else f'{values[i,j]:.2f}'
            ax[2].text(j,i,number,ha='center',va='center',fontsize=8.2,
                        color='white' if values[i,j]>1.8 else P['navy'])
    ax[2].set_xlabel('Mean cost gap (%)')
    ax[2].set_title('(c) Cost gap by subset',loc='left',pad=13)
    fig.subplots_adjust(left=.075,right=.99,top=.79,bottom=.28,wspace=.80)
    save(fig,'fig6_lifecycle_scalar_paths')
    (DERIVED/'scalar_switch_and_cost_gaps.json').write_text(json.dumps(dict(
        switch_mean_lifetime=float(switch),intervals=intervals,mean_gaps_pct=values.tolist()),indent=2),encoding='utf-8')


def information_recovery():
    info=pd.read_csv(PUB/'results_information_audit'/'information_audit_configurations.csv')
    curve=pd.read_csv(PUB/'results_information_audit'/'historical_threshold_curve.csv')
    masks=[info.censor_pct<90,(info.censor_pct>=90)&(info.censor_pct<100),info.censor_pct>=100]
    retained=[int((info.loc[m,'relative_gap_pct'].abs()<1e-10).sum()) for m in masks]
    increased=[int(m.sum())-k for m,k in zip(masks,retained)]
    assert retained==[91,38,1] and increased==[0,0,14]
    fig,ax=plt.subplots(1,2,figsize=(7.25,2.8))
    y=np.arange(3)
    ax[0].barh(y,retained,color=P['teal'],height=.52,label='Full optimum retained')
    ax[0].barh(y,increased,left=retained,color=P['orange'],height=.52,label='Positive cost penalty')
    for i,(a,b) in enumerate(zip(retained,increased)):
        if a>5:ax[0].text(a/2,i,str(a),ha='center',va='center',color='white',weight='bold')
        else:ax[0].text(a+1.0,i-.24,f'{a} retained',fontsize=8.3,ha='left',color=P['teal'])
        if b:ax[0].text(a+b/2,i,str(b),ha='center',va='center',color='white',weight='bold')
    ax[0].set_yticks(y,['<90% censored','90–<100%','100% censored']);ax[0].invert_yaxis()
    ax[0].set(xlabel='Number of configurations',xlim=(0,100))
    ax[0].set_title('(a) Optimum retention by censoring',loc='left',pad=32)
    ax[0].legend(loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=1,frameon=False,borderaxespad=0,fontsize=8.1)
    ax[1].plot(curve.historical_margin,curve.identified_objective,color=P['navy'],lw=1.7,label='Minimum on identified range')
    ax[1].axhline(curve.complete_safe_objective.iloc[0],color=P['orange'],ls='--',lw=1.1,label='Full risk-feasible minimum')
    star=float(curve.complete_safe_margin.iloc[0])
    ax[1].axvline(star,color=P['gray'],ls=':',lw=.8)
    ax[1].annotate(f'Full-data optimum\nmargin = {star:.3f}',xy=(star,.5804),xytext=(.5,.606),
                   fontsize=8.3,color=P['navy'],arrowprops=dict(arrowstyle='->',color=P['gray'],lw=.8))
    ax[1].fill_between(curve.historical_margin,curve.complete_safe_objective,curve.identified_objective,color=P['navy'],alpha=.09)
    ax[1].set(xlabel=r'Historical threshold $h$',ylabel='Cal event cost / 100 service cycles')
    ax[1].set_title('(b) Advancing the historical alarm',loc='left',pad=32)
    ax[1].legend(loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=1,frameon=False,borderaxespad=0,fontsize=8.1)
    fig.subplots_adjust(left=.17,right=.98,bottom=.24,top=.69,wspace=.58)
    save(fig,'fig4_censoring_identification')


def event_cost():
    primary=pd.read_csv(PUB/'results_retirement_event'/'retirement_event_primary.csv')
    primary=primary[primary.scenario=='Tune_ERM']
    boot=pd.read_csv(PUB/'results_retirement_event'/'retirement_event_paired_bootstrap.csv')
    fig=plt.figure(figsize=(7.25,2.6))
    ax=fig.add_axes([.085,.25,.35,.54]);right=fig.add_axes([.60,.25,.37,.54])
    names=['Historical','Retirement_floor','Retirement_event_opt'];labels=['Historical','Risk floor','Exact search']
    for j,(head,color) in enumerate([('Ridge',P['navy']),('Linex',P['orange'])]):
        values=primary[primary['head']==head].set_index('method').loc[names].event_cost
        ax.bar(np.arange(3)+(j-.5)*.34,values,width=.31,color=color,label=head)
    ax.set_xticks(range(3),labels);ax.set_ylabel('Event cost / 100 service cycles');ax.set_ylim(0,.95)
    ax.set_title('(a) Event costs after policy selection',loc='left',pad=31,fontsize=9)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,frameon=False,borderaxespad=0,fontsize=8.5)
    for i,head in enumerate(['Ridge','Linex']):
        row=boot[(boot['head']==head)&(boot.comparator=='Retirement_floor')].iloc[0]
        color=P['navy'] if head=='Ridge' else P['orange']
        right.errorbar(row.delta_event_cost,i,xerr=[[row.delta_event_cost-row.ci95_low],[row.ci95_high-row.delta_event_cost]],
                       fmt='o',color=color,lw=1.3,capsize=3,ms=5)
        right.text(.01,i-.21,f'{row.delta_event_cost:+.4f} [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]',
                   transform=right.get_yaxis_transform(),ha='left',va='bottom',fontsize=8.1,color=color)
    right.axvline(0,color=P['gray'],ls='--',lw=.9)
    right.set(ylim=(1.45,-.55),xlim=(-.052,.008),xlabel='Exact search − risk floor: cost difference')
    right.set_yticks([0,1],['Ridge','Linex'])
    right.set_title('(b) Paired differences and uncertainty',loc='left',pad=31,fontsize=9)
    right.text(0,1.04,'Dot: difference    Line: 95% bootstrap interval',transform=right.transAxes,
               fontsize=8.2,va='bottom',color=P['gray'])
    save(fig,'fig5_retirement_only_cost')


if __name__=='__main__':
    mechanism();scalar_recovery();information_recovery();event_cost()
    print('Redrew four figures with explicit data and uncertainty semantics.')
