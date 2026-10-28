"""Make manuscript tables and vector figures from the independently rerun CSVs."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import expit
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / 'Experiments' / '公开基准' / 'results_censored_lifecycle'
FIG = BASE / 'Manuscript' / 'figures'
sys.path.insert(0, str(BASE / 'Experiments' / '代码'))
import run_public_benchmark as rpb

P = dict(navy='#253E5C', teal='#167D8D', orange='#C56D32', purple='#84619B', gray='#7A858B', light='#EDF2F4')
STYLE={'font.family':'DejaVu Sans','font.size':9.2,'axes.labelsize':9.2,
                     'axes.titlesize':10,'axes.titleweight':'bold','xtick.labelsize':8.9,
                     'ytick.labelsize':8.9,'legend.fontsize':8.8,'pdf.fonttype':42,
                     'ps.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.7,'savefig.dpi':360,'axes.grid':False}
plt.rcParams.update(STYLE)
cfg = pd.read_csv(DATA / 'lifecycle_identification.csv')
primary = pd.read_csv(DATA / 'lifecycle_primary.csv')
pooled = pd.read_csv(DATA / 'lifecycle_pooled_by_seed.csv')
boot = pd.read_csv(DATA / 'lifecycle_paired_bootstrap.csv')
states = pd.read_csv(DATA / 'lifecycle_candidate_states.csv')
paths = pd.read_csv(DATA / 'lifecycle_scalar_paths.csv')


def subset(frame, dataset='FD002', seed=20261023, head='Linex', scenario='Tune_Q85'):
    return frame[(frame.dataset == dataset) & (frame.seed == seed) & (frame['head'] == head) & (frame.scenario == scenario)]


def save(figure, name):
    figure.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.03)
    figure.savefig(FIG / (name + '.png'), bbox_inches='tight', pad_inches=.03)
    plt.close(figure)


def scalar_figures():
    figure, ax = plt.subplots(1, 3, figsize=(7.25, 2.42), gridspec_kw={'width_ratios':[1.08,1.05,1.12]})
    case = subset(cfg, head='Ridge', scenario='Tune_ERM').iloc[0]
    path = subset(paths, head='Ridge', scenario='Tune_ERM')
    n = int(case.n_cal)
    case_states=subset(states,head='Ridge',scenario='Tune_ERM')
    # Construct the exact lower envelope of affine candidate costs. Ordered slopes
    # permit a hull scan; intersection points are the exact decision transitions.
    hull=[]
    for _,state in case_states.iloc[::-1].iterrows():
        slope=4.5/state.service_sum
        intercept=100*state.event_cost_sum/state.service_sum-4.5
        line=(slope,intercept,float(state.margin))
        while hull and abs(hull[-1][0][0]-slope)<1e-15:
            if intercept <= hull[-1][0][1]+1e-14:hull.pop()
            else:break
        if hull and abs(hull[-1][0][0]-slope)<1e-15:continue
        crossing=-math.inf
        while hull:
            previous,begin=hull[-1]
            crossing=(intercept-previous[1])/(previous[0]-slope)
            if crossing<=begin:hull.pop()
            else:break
        hull.append((line,crossing if hull else -math.inf))
    x_path=[case.S_T_low]
    y_path=[float(path.iloc[0].selected_margin)]
    for line,begin in hull:
        if case.S_T_low < begin < case.S_T_up:
            x_path.append(begin);y_path.append(line[2])
    x_path.append(case.S_T_up);y_path.append(y_path[-1])
    ax[0].step(np.asarray(x_path)/n,y_path,where='post',color=P['teal'],lw=1.8)
    for x, color, label in [(case.S_T_low/n, P['orange'], r'$s_-/n$'),
                            (case.S_T_true/n, P['navy'], r'$S_T/n$'),
                            (case.S_T_up/n, P['gray'], r'$s_+/n$')]:
        ax[0].axvline(x, color=color, ls='--', lw=.9)
        ax[0].text(x, 1.01, label, color=color, ha='center', transform=ax[0].get_xaxis_transform())
    ax[0].set(xlabel=r'Lifetime scalar $s/n$ (cycles)', ylabel=r'Optimal margin $\lambda_l(s)$')
    ax[0].set_title('(a) Threshold path', loc='left', pad=21)
    ax[0].text(.98,.95,'FD002 · Ridge\nTune-ERM history', transform=ax[0].transAxes, ha='right',va='top',fontsize=8.8)
    names=['Event cost','Observable life','Interval minimax','Mean scalar']
    counts=[int(cfg.je_matches_ray.sum()),int(cfg.low_matches_ray.sum()),int(cfg.mm_matches_ray.sum()),int(cfg.mean_matches_ray.sum())]
    colors=[P['gray'],P['teal'],P['purple'],P['navy']]
    y=np.arange(4)
    ax[1].barh(y, counts, height=.55,color=colors)
    ax[1].set_yticks(y,names)
    ax[1].invert_yaxis()
    ax[1].set(xlim=(0,164),xlabel='Same-ray minima attained / 144')
    for i,k in enumerate(counts):ax[1].text(k+2,i,str(k),va='center',fontweight='bold',fontsize=8)
    ax[1].set_title('(b) Optimum recovery',loc='left',pad=21)
    censor=100*cfg.n_censored/cfg.n_cal
    event=100*(cfg.cal_jl_je/cfg.oracle_ray_jl-1)
    low=100*(cfg.cal_jl_low/cfg.oracle_ray_jl-1)
    ax[2].scatter(censor,event,s=12,c=P['gray'],alpha=.55,label='Event cost',marker='o',linewidths=0)
    ax[2].scatter(censor,low,s=17,c=P['teal'],alpha=.8,label='Observable life',marker='x',linewidths=.7)
    ax[2].axhline(0,color='#B8C2C8',lw=.6,zorder=0)
    ax[2].set(xlabel='Preventive censoring (%)',ylabel='Excess Cal lifecycle cost (%)',xlim=(0,104))
    ax[2].set_title('(c) Cost recovery',loc='left',pad=21)
    ax[2].legend(loc='upper left',frameon=False,handlelength=1)
    figure.subplots_adjust(wspace=.71,bottom=.22,top=.78,left=.07,right=.99)
    save(figure,'fig6_lifecycle_scalar_paths')


def cost_figure():
    figure,ax=plt.subplots(1,3,figsize=(7.25,2.7),gridspec_kw={'width_ratios':[1,1,.95]})
    methods=['Retirement_floor','Event_sweep_existing','Censored_lifecycle_low','Complete_lifecycle_same_ray']
    labels=['Calibrated floor','Event cost','Observable life','Complete same-ray']
    colors=[P['gray'],P['orange'],P['teal'],P['navy']]
    for hi,head in enumerate(['Linex','Ridge']):
        data=primary[(primary['head']==head)&(primary.scenario=='Tune_Q70')].set_index('method')
        ys=np.arange(4)+hi*5
        for i,m in enumerate(methods):
            row=data.loc[m]
            ax[0].plot(row.j_life,ys[i],'o',color=colors[i],ms=4.5)
            ax[0].text(row.j_life+.006,ys[i]-.07,f'{int(row.failures)}',fontsize=8.8,color=colors[i])
            ax[1].plot(row.waste_cycles,ys[i],'o',color=colors[i],ms=4.5)
        ax[0].text(.98,1-(hi*5+.4)/9.8,head,transform=ax[0].transAxes,ha='right',va='top',fontweight='bold',fontsize=8)
    yy=np.r_[np.arange(4),np.arange(4)+5]
    for i in range(2):
        ax[i].set_yticks(yy,labels*2 if i==0 else ['']*8)
        ax[i].set_ylim(8.8,-1)
        ax[i].axhline(4,color='#D3DADF',lw=.7)
    ax[0].set(xlabel=r'Lifecycle cost $J_l$',xlim=(1.075,1.225))
    ax[1].set(xlabel='Mean unused life (cycles)',xlim=(15,22.6))
    ax[0].set_title('(a) Lifecycle cost',loc='left',pad=12)
    ax[1].set_title('(b) Retained service',loc='left',pad=12)
    comparisons=[('Linex','Event_sweep_existing','Linex / event'),('Linex','Retirement_floor','Linex / floor'),
                 ('Ridge','Event_sweep_existing','Ridge / event'),('Ridge','Retirement_floor','Ridge / floor')]
    for i,(head,comp,label) in enumerate(comparisons):
        row=boot[(boot['head']==head)&(boot.scenario=='Tune_Q70')&(boot.treatment=='Censored_lifecycle_low')&(boot.comparator==comp)].iloc[0]
        color=P['teal'] if head=='Linex' else P['navy']
        ax[2].errorbar(row.delta_j_life,i,xerr=[[row.delta_j_life-row.ci95_jl_low],[row.ci95_jl_high-row.delta_j_life]],
                       fmt='o',color=color,capsize=2,ms=4,lw=1)
    ax[2].axvline(0,color=P['gray'],lw=.8,ls='--')
    ax[2].set_yticks(range(4),[r[2] for r in comparisons])
    ax[2].invert_yaxis()
    ax[2].set(xlabel=r'Cost difference $\Delta J_l$')
    ax[2].set_title('(c) Paired differences',loc='left',pad=12)
    figure.subplots_adjust(left=.19,right=.99,bottom=.22,top=.86,wspace=.60)
    save(figure,'fig7_lifecycle_cost_service')


def mechanism_figure():
    row=subset(cfg).iloc[0]
    ds='FD002'
    frame=rpb.load_dataset(BASE/'Experiments'/'公开基准'/'raw'/f'train_{ds}.txt')
    parts,_=rpb.split_frame(frame,20261023)
    core=rpb.load_core(BASE/'Experiments'/'代码'/'decision_calibrated_pdm_suite.py')
    plt.rcParams.update(STYLE)
    enc=core.CausalMonotoneEncoder(True,True).fit(parts['fit'])
    x={k:enc.transform(v)[0] for k,v in parts.items()}
    model=core.PrognosticHead().fit(x['fit'],parts['fit'].true_rul.to_numpy(),kind='linex',reg=14.,focus_window=True)
    p={k:model.predict(v,parts[k],causal_smooth=True) for k,v in x.items()}
    bias=float(np.mean(p['tune']-parts['tune'].true_rul.to_numpy()))
    p={k:np.clip(v-.85*bias,0,125) for k,v in p.items()}
    sd=max(1e-6,float(np.std(p['tune']-parts['tune'].true_rul.to_numpy())))
    scale={k:sd*(.3+.7*expit((37-v)/9)) for k,v in p.items()}
    seq=rpb.unit_sequences(parts['cal'],p['cal'],scale['cal'])
    unit=next(u for u in seq if u['unit_id']==42)
    t=unit['cycles'];score=(unit['point']-15)/unit['scale']
    historical=t[np.flatnonzero(score<=row.h)[0]]
    eventalarm=t[np.flatnonzero(score<=row.lam_je)[0]]
    tau_life=t[np.flatnonzero(score<=row.lam_jl_low)[0]]
    observed=t<=historical
    sample=pd.DataFrame({'cycle':t,'score':score,'observed_prefix':observed.astype(int)})
    sample.to_csv(BASE/'Experiments'/'derived'/'lifecycle_mechanism_trace.csv',index=False)
    metadata=dict(dataset=ds,seed=20261023,head='Linex',phase='Cal',history='Tune_Q85',
                  example_unit_id=42,n_cal=int(row.n_cal),n_censored=int(row.n_censored),
                  h=float(row.h),risk_floor=float(row.L_R),selected_lifecycle_margin=float(row.lam_jl_low),
                  selected_event_margin=float(row.lam_je),historical_alarm=float(historical),
                  lifecycle_alarm=float(tau_life),event_alarm=float(eventalarm),
                  scalar_lower=float(row.S_T_low),scalar_reference=float(row.S_T_true),
                  scalar_stress_upper=float(row.S_T_up))
    (FIG/'mechanism_figure_data.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    figure=plt.figure(figsize=(7.25,4.35))
    flow=figure.add_axes([.02,.70,.96,.27]);flow.set(xlim=(0,1),ylim=(0,1));flow.axis('off')
    flow.text(0,1.02,'(a) A decision layer from recorded history to lifecycle choice',fontweight='bold',fontsize=9.2)
    boxes=[('Frozen prognosis',r'$a_{u,t}=(f_{u,t}-\ell)/s_{u,t}$'),
           ('Observed retirement',r'prefix, $U_u^h$, $Z_u^h$, $h$'),
           ('Scalar cost frontier',r'$g(\lambda;s)=A_\lambda+B_\lambda s$'),
           ('Risk-feasible choice',r'$\lambda_l(s_-)$ / endpoint regret')]
    for i,(title,detail) in enumerate(boxes):
        x0=.015+i*.248
        patch=FancyBboxPatch((x0,.30),.225,.48,boxstyle='round,pad=.01,rounding_size=.025',
                             facecolor=P['light'],edgecolor=P['teal'] if i>=2 else P['navy'],lw=.9)
        flow.add_patch(patch)
        flow.text(x0+.1125,.63,title,ha='center',va='center',fontweight='bold',fontsize=8.8)
        flow.text(x0+.1125,.42,detail,ha='center',va='center',fontsize=8.6)
        if i<3:flow.add_patch(FancyArrowPatch((x0+.228,.54),(x0+.25,.54),arrowstyle='-|>',mutation_scale=8,color=P['gray']))
    flow.text(.5,.05,r'Nested alarms reconstruct outcomes; $W=T-U$ concentrates latent lifetime into $s=\sum_uT_u$.',
              ha='center',fontsize=8.6,color=P['navy'])
    left=figure.add_axes([.08,.15,.39,.43]);right=figure.add_axes([.60,.15,.38,.43])
    left.plot(t[observed],score[observed],color=P['navy'],lw=1.4,label='Recorded prefix')
    left.plot(t[~observed],score[~observed],color=P['gray'],ls='--',lw=1,label='Complete-data reference')
    left.axvspan(historical,unit['T'],facecolor='#DBE1E5',alpha=.42)
    left.axhline(row.h,color=P['gray'],lw=.8,ls=':',label='Historical margin')
    left.axhline(row.lam_je,color=P['orange'],lw=.9,ls='--',label='Event-optimal margin')
    left.axvline(historical,color=P['gray'],lw=.7,ls=':')
    left.scatter([eventalarm,historical],[row.lam_je,row.h],s=18,c=[P['orange'],P['navy']],zorder=5)
    left.set(xlabel='Inspection cycle',ylabel='Prognostic score',xlim=(max(1,eventalarm-70),unit['T']))
    left.set_title('(b) Nested alarms in the recorded prefix',loc='left',fontsize=8.5,pad=10)
    left.legend(frameon=False,fontsize=8.6,loc='lower left',handlelength=1.7)
    st=subset(states)
    xx=np.linspace(row.S_T_low,row.S_T_up,250)
    n=int(row.n_cal)
    allvals=100*(st.event_cost_sum.to_numpy()[:,None]+.045*xx[None,:])/st.service_sum.to_numpy()[:,None]-4.5
    candidates=[(row.L_R,'Risk floor / observable life',P['teal']),
                (row.lam_je,'Event-cost policy',P['orange'])]
    for lam,label,color in candidates:
        i=int(np.argmin(abs(st.margin.to_numpy()-lam)))
        right.plot(xx/n,allvals[i],color=color,lw=1.6,label=label)
    right.plot(xx/n,allvals.min(axis=0),color=P['navy'],ls='--',lw=1,label='Optimal cost envelope')
    right.axvline(row.S_T_true/n,color=P['gray'],ls=':',lw=.9)
    right.set(xlabel=r'Lifetime scalar $s/n$ (cycles)',ylabel='Cal lifecycle cost')
    right.set_title('(c) A scalar lifecycle-cost frontier',loc='left',fontsize=8.5,pad=10)
    right.legend(frameon=False,fontsize=8.6,loc='upper left',handlelength=1.6)
    right.text(.98,.04,'52 Cal engines; 48 censored\nOptimal margin stable for all '+r'$s\geq s_-$',
               transform=right.transAxes,ha='right',fontsize=8.6,color=P['teal'])
    save(figure,'fig1_decision_framework')


METHODS=[('Historical','Historical'),('Retirement_floor','Risk floor'),('Event_sweep_existing','Event cost'),
         ('Censored_lifecycle_low','Observable life'),('Censored_lifecycle_minimax','Interval minimax'),
         ('Censored_lifecycle_mean','Mean scalar'),('Complete_lifecycle_same_ray','Complete same-ray'),
         ('Complete_lifecycle_full_pool_cal','Complete full-ray')]


def tables():
    out=[r'\section{Censored-record lifecycle comparisons}',r'\label{app:lifecycle_results}',
         r'\subsection{Primary replay by historical rule}',
         'The tables report every lifecycle policy for the primary replay. Q70 and Q85 use the Tune critical-score percentiles; ERM uses the Tune lifecycle optimum. Each head--history comparison uses the same 286 Test engines.']
    for head in ['Linex','Ridge']:
        out += [r'\par\begingroup\small\setlength{\tabcolsep}{2.8pt}',
                r'\captionof{table}{Primary '+head+r' lifecycle policies. $W$ is mean unused natural life in cycles.}',
                r'\label{tab:lifecyclefullprimary}' if head=='Linex' else r'\label{tab:lifecycleridgeprimary}',
                r'\begin{tabular}{llrrr}',r'\toprule',r'History & Rule & $J_l$ & Fail. & $W$\\',r'\midrule']
        for scenario,label in [('Tune_Q70','Q70'),('Tune_Q85','Q85'),('Tune_ERM','ERM')]:
            for i,(method,name) in enumerate(METHODS):
                row=primary[(primary['head']==head)&(primary.scenario==scenario)&(primary.method==method)].iloc[0]
                out.append(f'{label if i==0 else ""} & {name} & {row.j_life:.4f} & {int(row.failures)} & {row.waste_cycles:.2f}'+r'\\')
            out.append(r'\midrule' if scenario!='Tune_ERM' else r'\bottomrule')
        out += [r'\end{tabular}',r'\par\endgroup',r'\medskip']
    out += [r'\subsection{Six-split descriptive sensitivity}',
            'The following tables report mean and sample standard deviation across the six fixed splits. Source engines recur across splits; the split summaries describe sensitivity of the fitted and calibrated rules.']
    for head in ['Linex','Ridge']:
        out += [r'\par\begingroup\small\setlength{\tabcolsep}{2.5pt}',
                r'\captionof{table}{Six-split '+head+r' lifecycle comparison.}',
                r'\begin{tabular}{llrr}',r'\toprule',r'History & Rule & $J_l$ mean $\pm$ SD & Fail. (\%)\\',r'\midrule']
        for scenario,label in [('Tune_Q70','Q70'),('Tune_Q85','Q85'),('Tune_ERM','ERM')]:
            for i,(method,name) in enumerate(METHODS):
                g=pooled[(pooled['head']==head)&(pooled.scenario==scenario)&(pooled.method==method)]
                out.append(f'{label if i==0 else ""} & {name} & ${g.j_life.mean():.4f}\\pm {g.j_life.std():.4f}$ & {g.fail_pct.mean():.2f}'+r'\\')
            out.append(r'\midrule' if scenario!='Tune_ERM' else r'\bottomrule')
        out += [r'\end{tabular}',r'\par\endgroup',r'\medskip']
    out += [r'\subsection{Paired primary lifecycle differences}',
            r'Each interval uses 4,000 paired, subset-stratified engine resamples conditional on the selected rules. The treatment is observable lifecycle selection at $s_-$. Q70 and Q85 give identical comparisons in the primary replay.']
    out += [r'\par\begingroup\small\setlength{\tabcolsep}{3pt}',
            r'\captionof{table}{Primary paired lifecycle differences for Q70/Q85.}',
            r'\begin{tabular}{llrr}',r'\toprule',r'Head & Comparator & $\Delta J_l$ & 95\% interval\\',r'\midrule']
    for head in ['Linex','Ridge']:
        for comparator,name in [('Event_sweep_existing','Event cost'),('Retirement_floor','Risk floor'),('Historical','Q70 historical')]:
            r=boot[(boot['head']==head)&(boot.scenario=='Tune_Q70')&(boot.treatment=='Censored_lifecycle_low')&(boot.comparator==comparator)].iloc[0]
            out.append(f'{head} & {name} & ${r.delta_j_life:.4f}$ & $[{r.ci95_jl_low:.4f},{r.ci95_jl_high:.4f}]$'+r'\\')
    out += [r'\bottomrule',r'\end{tabular}',r'\par\endgroup']
    (BASE/'Manuscript'/'Lifecycle_Results.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')


def tables_compact():
    out=[r'\section{Censored-record lifecycle comparisons}',r'\label{app:lifecycle_results}',
         r'\subsection{Primary replay by historical rule}',
         'The primary tables report all eight policies for each head and historical rule. Each comparison uses the same 286 Test engines. Costs are per 100 service cycles; $W$ is mean unused natural life in cycles.']
    for head in ['Linex','Ridge']:
        for scenario,label in [('Tune_Q70','Q70'),('Tune_Q85','Q85'),('Tune_ERM','Tune-ERM')]:
            out += [r'\par\medskip\noindent\begin{minipage}{\linewidth}\small\setlength{\tabcolsep}{3pt}',
                    r'\captionof{table}{Primary '+head+' policies under the '+label+r' history.}',
                    r'\label{tab:lifecyclefullprimary}' if head=='Linex' and scenario=='Tune_Q70' else '',
                    r'\begin{tabular}{lrrr}',r'\toprule',r'Rule & $J_l$ & Failures & $W$\\',r'\midrule']
            for method,name in METHODS:
                row=primary[(primary['head']==head)&(primary.scenario==scenario)&(primary.method==method)].iloc[0]
                out.append(f'{name} & {row.j_life:.4f} & {int(row.failures)} & {row.waste_cycles:.2f}'+r'\\')
            out += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}\par\medskip']
    out += [r'\subsection{Six-split descriptive sensitivity}',
            'Each table gives mean and sample standard deviation across the six fixed splits. Source engines recur across splits; these summaries describe sensitivity of the fitted and calibrated rules.']
    for head in ['Linex','Ridge']:
        for scenario,label in [('Tune_Q70','Q70'),('Tune_Q85','Q85'),('Tune_ERM','Tune-ERM')]:
            out += [r'\par\medskip\noindent\begin{minipage}{\linewidth}\small\setlength{\tabcolsep}{3pt}',
                    r'\captionof{table}{Six-split '+head+' policies under the '+label+r' history.}',
                    r'\begin{tabular}{lrr}',r'\toprule',r'Rule & $J_l$ mean $\pm$ SD & Failures (\%)\\',r'\midrule']
            for method,name in METHODS:
                g=pooled[(pooled['head']==head)&(pooled.scenario==scenario)&(pooled.method==method)]
                out.append(f'{name} & ${g.j_life.mean():.4f}\\pm {g.j_life.std():.4f}$ & {g.fail_pct.mean():.2f}'+r'\\')
            out += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}\par\medskip']
    out += [r'\subsection{Paired primary lifecycle differences}',
            r'Intervals use 4,000 paired, subset-stratified engine resamples conditional on the selected rules. Treatment is observable lifecycle selection at $s_-$. Q70 and Q85 give identical selected-policy comparisons.',
            r'\par\medskip\noindent\begin{minipage}{\linewidth}\small\setlength{\tabcolsep}{3pt}',
            r'\captionof{table}{Primary paired lifecycle differences under Q70/Q85.}',
            r'\begin{tabular}{llrr}',r'\toprule',r'Head & Comparator & $\Delta J_l$ & 95\% interval\\',r'\midrule']
    for head in ['Linex','Ridge']:
        for comparator,name in [('Event_sweep_existing','Event cost'),('Retirement_floor','Risk floor'),('Historical','Q70 historical')]:
            row=boot[(boot['head']==head)&(boot.scenario=='Tune_Q70')&(boot.treatment=='Censored_lifecycle_low')&(boot.comparator==comparator)].iloc[0]
            out.append(f'{head} & {name} & ${row.delta_j_life:.4f}$ & $[{row.ci95_jl_low:.4f},{row.ci95_jl_high:.4f}]$'+r'\\')
    out += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}\par\medskip']
    (BASE/'Manuscript'/'Lifecycle_Results.tex').write_text('\n'.join(out)+'\n',encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        scalar_figures()
        cost_figure()
        mechanism_figure()
    audit={name:dict(source='results_censored_lifecycle CSV files',width_inches=7.25) for name in
           ['fig1_decision_framework','fig6_lifecycle_scalar_paths','fig7_lifecycle_cost_service']}
    (FIG/'lifecycle_figure_provenance.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Rebuilt three lifecycle vector figures from archived numerical records.')
