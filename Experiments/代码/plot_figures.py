"""Redraw presentation figures from verified CSVs without changing numerical results.

The median-unit trajectory is regenerated using the unchanged experimental model.
No external paper PDFs or images are downloaded. All plots are ordinary matplotlib
vector graphics derived from numerical records.
"""
from pathlib import Path
import argparse,importlib.util,json,math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(*args, **kwargs):
        return nullcontext()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

P={'navy':'#213A5C','red':'#B33B36','teal':'#247D75','gold':'#B88622','gray':'#707981','purple':'#775C86','orange':'#BE693C'}
PLOT_STYLE={'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,
 'axes.titleweight':'bold','xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
 'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.7,'savefig.dpi':600,'axes.grid':False}
plt.rcParams.update(PLOT_STYLE)

def load_sample(suite_path):
    sp=importlib.util.spec_from_file_location('verified_suite_for_figures',suite_path)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);m._C_PAVA_LIB=False
    with threadpool_limits(limits=1):
        df=m.generate_synthetic_fleet('S-FD004',seed=33)
        parts=[df[df.unit_id<=60].copy(),df[(df.unit_id>60)&(df.unit_id<=90)].copy(),
               df[(df.unit_id>90)&(df.unit_id<=130)].copy(),df[df.unit_id>130].copy()]
        fit,tune,cal,test=parts
        enc=m.CausalMonotoneEncoder(True,True).fit(fit)
        xx=[enc.transform(p)[0] for p in parts]
        yf=fit.true_rul.to_numpy();yt=tune.true_rul.to_numpy();yc=cal.true_rul.to_numpy()
        ridge=m.PrognosticHead().fit(xx[0],yf,kind='ridge',reg=14.)
        head=m.PrognosticHead().fit(xx[0],yf,kind='linex',rho=10.,reg=14.,focus_window=True)
        pt,pc,pe=[head.predict(x,p,causal_smooth=True) for x,p in zip(xx[1:],parts[1:])]
        bias=np.mean(pt-yt)
        pt,pc,pe=[np.clip(p-.85*bias,0,125) for p in [pt,pc,pe]]
        sd=np.std(pt-yt);st,sc,se=[sd*(.3+.7*expit((37-p)/9)) for p in [pt,pc,pe]]
        B=[m.unit_critical_score(cal.iloc[pos].cycle.to_numpy(),pc[pos],sc[pos],float(cal.iloc[pos].max_cycle.iloc[0]))
           for _,pos in cal.groupby('unit_id',sort=False).indices.items()]
        lam_t=m.tune_differentiable_policy_margin(tune,pt,st)
        lam,_=m.choose_static_margin(B,alpha=.05,tuned_margin=lam_t)
        prc=ridge.predict(xx[2]);pre=ridge.predict(xx[3])
        scores=np.sort(prc-yc);q=max(0.,float(scores[math.ceil((len(scores)+1)*.95)-1]))
        lifetimes=test.groupby('unit_id').max_cycle.first().sort_values()
        uid=int(lifetimes.index[len(lifetimes)//2]);mask=test.unit_id.to_numpy()==uid
        g=test[mask]
        sample=pd.DataFrame({'cycle':g.cycle.to_numpy(),'T_u':g.max_cycle.to_numpy(),
          'true_uncapped_rul':g.true_rul_raw.to_numpy(),'point_forecast':pe[mask],
          'unit_calibrated_statistic':pe[mask]-lam*se[mask],
          'cycle_CP_bound':pre[mask]-q})
        alarm=sample.loc[sample.unit_calibrated_statistic<=15,'cycle']
        meta={'scenario':'S-FD004','seed':33,'unit_id':uid,'T_u':int(g.max_cycle.iloc[0]),
           'tau_static':int(alarm.iloc[0]) if len(alarm) else None,'margin_static':float(lam),
           'generation':'Unchanged verified model, Python PAVA fallback; only representative trajectory regenerated.'}
        return sample,meta

def clean(ax):
    ax.set_axisbelow(True);ax.grid(axis='x',color='#DDE1E5',lw=.6)

def save(fig,name,folders,checks):
    fig.canvas.draw()
    for folder in folders:
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',pad_inches=.10)
        fig.savefig(folder/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.10)
    checks.append({'figure':name,'natural_width_in':float(fig.get_figwidth()),
       'natural_height_in':float(fig.get_figheight()),'primary_format':'vector PDF',
       'preview_png_dpi':600,'font':'Arial with STIX math','base_font_pt':9.5})
    plt.close(fig)

def draw(results,folders,suite_path,sample=None):
    r=Path(results);folders=[Path(p) for p in folders];checks=[]
    unit=pd.read_csv(r/'unit_policy_b1_b10_per_unit.csv')
    seed=pd.read_csv(r/'unit_policy_b1_b10_per_seed.csv')
    summary=pd.read_csv(r/'unit_policy_b1_b10_summary.csv')
    if sample is None:sample,meta=load_sample(suite_path)
    else:sample,meta=sample
    plt.rcParams.update(PLOT_STYLE)
    derived=r.parent/'derived';derived.mkdir(parents=True,exist_ok=True)
    sample.to_csv(derived/'figure1_median_unit_replay.csv',index=False)
    (derived/'figure1_median_unit_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    fig=plt.figure(figsize=(6.8,7.7));gs=fig.add_gridspec(3,1,height_ratios=[2.0,2.7,1.65],hspace=.65)
    ax=fig.add_subplot(gs[0]);ax.set_xlim(0,100);ax.set_ylim(0,100);ax.axis('off')
    ax.set_title('(a) Data partition and observable maintenance outcomes',loc='left',pad=10)
    for x,color,label in [(0,P['navy'],'Fit: 60 units\nEncoder\nand head'),(25.5,P['teal'],'Tune: 30 units\nBias, scale\nand margin'),
                           (51,P['gold'],'Calibrate: 40\nUnit-score\nquantile'),(76.5,P['red'],'Test: 50 units\nPolicy\nreplay')]:
        ax.add_patch(FancyBboxPatch((x,60),23,38,boxstyle='round,pad=.5',facecolor=color,edgecolor='white'))
        ax.text(x+11.5,79,label,ha='center',va='center',fontsize=9,color='white')
    for y,st,col in [(43,'Preventive: alarm → lead time → replacement; natural failure time is hidden',P['teal']),
                     (18,'Breakdown: alarm + lead time ≥ failure time; the failure event is observed',P['red'])]:
        ax.text(1,y,st,fontsize=9,color=col,ha='left',va='center')

    ax=fig.add_subplot(gs[1]);cy=sample.cycle
    for key,label,col,style in [('true_uncapped_rul','Natural remaining life',P['navy'],'-'),
      ('point_forecast','Point forecast',P['teal'],'--'),('unit_calibrated_statistic','Unit-calibrated statistic',P['red'],'-'),
      ('cycle_CP_bound','Cycle-level conformal bound',P['gray'],':')]:
        ax.plot(cy,sample[key],label=label,color=col,ls=style,lw=1.25)
    ax.axhline(15,color='black',ls='-.',lw=.8)
    tau=meta['tau_static'];ax.axvline(tau,color=P['red'],ls=':',lw=.8)
    alarm_y=float(sample.loc[sample.cycle==tau,'unit_calibrated_statistic'].iloc[0])
    ax.scatter([tau],[alarm_y],s=22,color=P['red'],zorder=5)
    ax.set_title(f"(b) Median-lifetime test unit {meta['unit_id']}: alarm {tau}, replacement {tau+15}",loc='left',pad=34)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,frameon=False,borderaxespad=0,columnspacing=1.0)
    ax.set_xlabel('Operating cycle');ax.set_ylabel('Remaining cycles');ax.set_xlim(1,meta['T_u']);clean(ax)

    ax=fig.add_subplot(gs[2]);clips={}
    for key,label,col in [('B5','Cycle-level conformal',P['gray']),('B10','Online unit policy',P['red'])]:
        tauarr=np.where(unit[key+'_tau']<0,np.inf,unit[key+'_tau'])
        buff=unit.T_u-(tauarr+15);clips[key]=int(((buff< -25)|(buff>60)).sum())
        ax.hist(np.clip(buff,-25,60),bins=np.linspace(-25,60,22),alpha=.60,color=col,label=label,density=True)
    ax.axvline(0,color='black',ls='--',lw=.8)
    ax.set_title('(c) Lead-time safety buffer across 1,000 test units',loc='left',pad=8)
    ax.set_xlabel('Failure time − (alarm time + 15 cycles)');ax.set_ylabel('Density')
    ax.legend(loc='upper right',frameon=False);clean(ax)
    meta['histogram_clipped_units']=clips
    save(fig,'fig1_framework_and_trajectories',folders,checks)

    labels=['B2 Cost-tuned Ridge','B3 Pinball + conformal','B5 Cycle-level conformal','B6 Cycle-level online',
            'B7 Smooth-cost tuning','B8 Pure unit conformal','B9 Static unit policy','B10 Online unit policy']
    codes=['B2','B3','B5','B6','B7','B8','B9','B10'];colors=[P['orange'],P['gray'],P['gray'],P['gray'],P['gold'],P['purple'],P['teal'],P['red']]
    methods=[next(m for m in summary.method if m.startswith(code+':')) for code in codes]
    fig=plt.figure(figsize=(6.8,7.2));gs=fig.add_gridspec(2,2,width_ratios=[4.2,1.6],height_ratios=[1,1],hspace=.52,wspace=.16)
    ax=fig.add_subplot(gs[0,0]);side=fig.add_subplot(gs[0,1],sharey=ax);side.axis('off')
    arrays=[seed[seed.method==m].sort_values(['scenario','seed']).j_life_rate.to_numpy() for m in methods]
    y=np.arange(8);bp=ax.boxplot(arrays,positions=y,vert=False,widths=.55,patch_artist=True,showfliers=False)
    for box,col in zip(bp['boxes'],colors):box.set_facecolor(col);box.set_alpha(.65)
    rng=np.random.default_rng(42)
    for j,(vec,col) in enumerate(zip(arrays,colors)):
        ax.scatter(vec,j+rng.uniform(-.13,.13,len(vec)),s=9,facecolor=col,edgecolor='black',lw=.25,zorder=3)
        diff=vec-arrays[-1];h=stats.t.ppf(.975,19)*diff.std(ddof=1)/np.sqrt(20)
        side.text(0,j,'Reference' if j==7 else f'{diff.mean():+.3f} ± {h:.3f}',va='center',fontsize=9)
    side.set_title('Difference vs. B10\n95% t interval',fontsize=9,pad=10)
    ax.set_yticks(y,labels);ax.invert_yaxis()
    all_cost=np.concatenate(arrays);span=np.ptp(all_cost)
    ax.set_xlim(all_cost.min()-.035*span,all_cost.max()+.035*span)
    ax.set_xlabel('Lifecycle cost per 100 useful cycles');ax.set_title('(a) Cost across 20 scenario–seed runs',loc='left',pad=10);clean(ax)
    ax=fig.add_subplot(gs[1,0]);side=fig.add_subplot(gs[1,1],sharey=ax);side.axis('off')
    ss=summary.set_index('method').loc[methods]
    for j,(m,col) in enumerate(zip(methods,colors)):
        row=ss.loc[m];p=float(row.fail_pct);up=float(row.clopper_pearson_ucb95_pct)
        ax.errorbar(p,j,xerr=[[0],[up-p]],fmt='o',color=col,ecolor=col,capsize=3,ms=5,lw=1.2)
        side.text(0,j,f'{float(row.waste_cycles_mean):.2f}',va='center',fontsize=9)
    ax.axvline(5,color=P['red'],ls='--',lw=1)
    ax.set_xlim(-.2,6.6);ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.set_xlabel('Failure proportion (%) and descriptive upper bound')
    ax.set_title('(b) Failure incidence across 1,000 test units',loc='left',pad=10)
    side.set_title('Mean unused life\n(cycles)',fontsize=9,pad=10);clean(ax)
    fig.subplots_adjust(left=.27,right=.98,top=.93,bottom=.08)
    save(fig,'fig2_benchmark_comparison',folders,checks)

    margin=pd.read_csv(r/'margin_controls_summary.csv');drift=pd.read_csv(r/'drift_scaling_summary.csv')
    paired=pd.read_csv(r/'attribution_paired_tests.csv').set_index('comparison')
    fig=plt.figure(figsize=(6.8,8.1));gs=fig.add_gridspec(3,1,height_ratios=[2.7,2.1,1.7],hspace=.72)
    ax=fig.add_subplot(gs[0]);values=[];errs=[];labs=[]
    for predictor in ['ridge','linex']:
        for pol,label in [('cycle_CP','Cycle CP'),('unit_CP_no_tune_floor','Pure unit CP'),
                          ('tuned_margin_only','Tuning only'),('unit_CP_plus_tune_floor','Tuning + unit CP')]:
            row=margin[(margin.predictor==predictor)&(margin.policy==pol)].iloc[0]
            values.append(row.j_life_mean);errs.append(row.j_life_sd);labs.append(predictor.title()+': '+label)
    ax.barh(np.arange(8),values,xerr=errs,color=[P['gray'],P['purple'],P['gold'],P['navy']]*2,capsize=2,height=.55)
    ax.set_yticks(np.arange(8),labs);ax.invert_yaxis();ax.axhline(3.5,color='#999999',ls=':',lw=.7)
    ax.set_xlim(0,1.8);ax.set_xlabel('Lifecycle cost per 100 useful cycles')
    ax.set_title('(a) Separate cost tuning from the conformal floor',loc='left',pad=9);clean(ax)
    ax=fig.add_subplot(gs[1])
    for col,label,c,marker in [('j_life_static','Original static',P['gray'],'s'),('j_life_matched_static','Matched static',P['navy'],'D'),
                              ('j_life_online','Online event',P['red'],'o')]:
        ax.plot(drift.drift_rate,drift[col],marker=marker,color=c,label=label,lw=1.25,ms=4)
    ax.set_xlabel('Injected prediction-bias severity');ax.set_ylabel('Lifecycle cost rate')
    ax.set_title('(b) Online updates with matched initial conservatism',loc='left',pad=27)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,borderaxespad=0)
    clean(ax)
    ax=fig.add_subplot(gs[2]);keys=['L1 minus L4','A0 minus A1','A3 minus A0','A5 minus A0','A4 minus A0','A6 minus A0']
    sub=paired.loc[keys];md=sub.mean_difference.to_numpy()
    ax.errorbar(md,np.arange(6),xerr=np.array([md-sub.ci95_low.to_numpy(),sub.ci95_high.to_numpy()-md]),fmt='o',color=P['navy'],ecolor=P['red'],capsize=3,ms=4)
    ax.set_yticks(np.arange(6),['Quadratic − Linex','Online − static','Remove Linex','Remove smoothing','Remove window weight','Remove monotone state'])
    ax.invert_yaxis();ax.axvline(0,color='black',ls='--',lw=.8)
    ax.set_title('(c) Five-seed paired differences with 95% t intervals',loc='left',pad=8)
    ax.set_xlabel('Difference in lifecycle cost rate');clean(ax)
    fig.subplots_adjust(left=.29,right=.97,top=.96,bottom=.07)
    save(fig,'fig3_drift_and_risk_calibration',folders,checks)

    censor=pd.read_csv(r/'censoring_observability_summary.csv');mis=pd.read_csv(r/'mismatch_stress_summary.csv')
    fig=plt.figure(figsize=(6.8,7.5));gs=fig.add_gridspec(3,1,height_ratios=[2,2.4,1.7],hspace=.65)
    ax=fig.add_subplot(gs[0]);labs=['C1 Oracle feedback','C2 Censored imputation','C3 Failures-only feedback','C4 Static unit policy','C5 Observable-event update']
    ax.barh(np.arange(5),censor.j_life_rate,color=[P['gray'],P['gold'],P['orange'],P['teal'],P['red']],height=.55)
    ax.set_yticks(np.arange(5),labs);ax.invert_yaxis();ax.set_xlim(0,1.6)
    ax.set_xlabel('Lifecycle cost per 100 useful cycles');ax.set_title('(a) Feedback under preventive right-censoring',loc='left',pad=8);clean(ax)
    ax=fig.add_subplot(gs[1]);x=np.arange(6);labels=['Negative\ndips','No\nshocks','Positive\nspikes','Two-sided\nshocks','True\nrecovery','Unseen\nregime 7']
    ax.bar(x-.18,mis.rmse_unconstrained_ridge,.36,color=P['gray'],label='Unconstrained Ridge')
    ax.bar(x+.18,mis.rmse_dc_pdm,.36,color=P['red'],label='Monotone + Linex')
    ax.set_xticks(x,labels);ax.set_ylabel('RUL RMSE (cycles)');ax.set_ylim(0,17)
    ax.set_title('(b) Prediction errors with per-mode refit',loc='left',pad=27)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False,borderaxespad=0)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#DDE1E5',lw=.6)
    ax=fig.add_subplot(gs[2]);ax.axis('off');ax.set_title('(c) Decision outcomes with per-mode refit',loc='left',pad=12)
    data=[[labels[i].replace('\n',' '),f'{mis.j_life_dc_pdm.iloc[i]:.3f}',f'{mis.fail_pct_dc_pdm.iloc[i]:.1f}%'] for i in range(6)]
    tab=ax.table(cellText=data,colLabels=['Mechanism','Lifecycle cost','Failures (%)'],bbox=[0,-.10,1,1.02],cellLoc='left',colLoc='left',colWidths=[.45,.30,.25])
    tab.auto_set_font_size(False);tab.set_fontsize(9)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor('#CDD3D8');cell.set_linewidth(.5)
        if i==0:cell.set_facecolor('#EEF1F4');cell.set_text_props(weight='bold')
    fig.subplots_adjust(left=.28,right=.97,top=.95,bottom=.045)
    save(fig,'fig4_ablation_and_pareto',folders,checks)

    # Figure 5: Economic Attribution Regimes & Semi-Empirical NIST Physical Trace Benchmarks
    # Zero bar clipping: uses separate subplots (c) and (d) with true native y-axes!
    h1_sum=pd.read_csv(r/'tuning_sample_scarcity_summary.csv')
    h2_sum=pd.read_csv(r/'commissioning_to_field_summary.csv')
    nist_sum=pd.read_csv(r/'nist_empirical_benchmark_summary.csv')
    fig=plt.figure(figsize=(6.8,8.2))
    gs=fig.add_gridspec(3,2,height_ratios=[1.9,2.4,2.0],hspace=.68,wspace=.28)

    ax=fig.add_subplot(gs[0,:])
    nt=h1_sum.n_tune.to_numpy()
    ax.plot(nt,h1_sum.j_erm_mean,marker='s',color=P['orange'],ls='--',lw=1.35,label='Cost-tuned margin only (ERM)')
    ax.plot(nt,h1_sum.j_pure_cp_mean,marker='^',color=P['purple'],ls=':',lw=1.25,label='Pure Unit-CP ($n_{\\mathrm{cal}}=40, \\alpha=0.05$)')
    ax.plot(nt,h1_sum.j_cp_mean,marker='o',color=P['red'],ls='-',lw=1.5,label='DC-PdM (Tune + Unit-CP, $\\alpha=0.05$)')
    for x_v,je,jc,fe,fc,p5 in zip(nt,h1_sum.j_erm_mean,h1_sum.j_cp_mean,h1_sum.fail_erm_pct,h1_sum.fail_cp_pct,h1_sum.p_t_5seed):
        y_top = (max(je, 0.917) + 0.045) if x_v == 30 else (je + 0.038)
        ax.text(x_v,y_top,f'{fe:.1f}% ($p_5$={p5:.2f})',ha='center',fontsize=7.6,color=P['orange'])
        ax.text(x_v,jc-.068,f'{fc:.1f}%',ha='center',fontsize=7.6,color=P['red'])
    ax.set_xticks(nt,[f'N={n}' for n in nt]);ax.set_xlim(4.2,31.8);ax.set_ylim(0.74,1.38)
    ax.set_xlabel('Tuning split size $N_{\\mathrm{tune}}$ (1,000 test trajectories; 5-seed-block $p_5$ shown)')
    ax.set_ylabel('Lifecycle cost rate $\\hat{J}_{\\mathrm{life}}$')
    ax.set_title('(a) Tuning-sample scarcity H1: mean reduction under 5-seed-block inference',loc='left',pad=24)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,borderaxespad=0,fontsize=8.0);clean(ax)

    ax=fig.add_subplot(gs[1,:])
    h2_labs=['ERM tune only (Comm.)','B7 Smooth-cost (Comm.)','Field-retuned ERM (40 Field)',
             'Field-retuned B7 (40 Field)','B5 Cycle-CP (40 Field)','Pure Unit-CP (40 Field)',
             'B9 DC-PdM Static (40 Field)','B10 DC-PdM Online (Field)']
    y2=np.arange(len(h2_labs));vals2=h2_sum.j_life_mean.to_numpy();sds2=h2_sum.j_life_sd.to_numpy();fails2=h2_sum.fail_pct_mean.to_numpy()
    cols2=[P['orange'],P['gold'],P['navy'],P['navy'],P['gray'],P['purple'],P['teal'],P['red']]
    ax.barh(y2,vals2,xerr=sds2,color=cols2,capsize=2,height=.58)
    for j,(v,s,f) in enumerate(zip(vals2,sds2,fails2)):
        ax.text(v+s+.04,j,f'{v:.3f} ({f:.1f}% fail)',va='center',fontsize=7.8)
    ax.set_yticks(y2,h2_labs);ax.invert_yaxis();ax.set_xlim(0,3.55)
    ax.set_xlabel('Lifecycle cost per 100 useful cycles (1,000 field test trajectories)')
    ax.set_title('(b) Commissioning-to-field shift H2: unadapted vs. equal-field-budget policies',loc='left',pad=8);clean(ax)

    # Panel (c): Semi-empirical NIST APR423 (disjoint 3 Fit / 3 Tune / 6 Cal / 3 Test, unclipped!)
    ax_c=fig.add_subplot(gs[2,0])
    g1=nist_sum[nist_sum.benchmark=='NIST_APR423_SemiEmpirical_Disjoint_125Splits'].iloc[:4]
    x_g1=np.arange(4);v_g1=g1.j_life_mean.to_numpy();f_g1=g1.fail_pct_mean.to_numpy()
    ax_c.bar(x_g1,v_g1,.55,color=[P['gray'],P['purple'],P['orange'],P['teal']])
    for i,(v,f) in enumerate(zip(v_g1,f_g1)):
        ax_c.text(i,v+.06,f'{v:.2f}\n({f:.1f}%)',ha='center',fontsize=7.3)
    ax_c.set_xticks(x_g1,['Cycle\nCP','Pure\nU-CP','ERM\nTune','DC-PdM\n(k=6)'],fontsize=7.8)
    ax_c.set_ylim(0,2.65);ax_c.set_ylabel('Cost rate $\\hat{J}_{\\mathrm{life}}$')
    ax_c.set_title('(c) NIST 8.4.2.3 semi-emp.\n(15 units, disjoint splits)',loc='left',pad=8,fontsize=9.2)
    ax_c.set_axisbelow(True);ax_c.grid(axis='y',color='#DDE1E5',lw=.6)

    # Panel (d): Semi-empirical NIST MPC61 (Run 1 -> Run 2, full unclipped axis up to 7.2!)
    ax_d=fig.add_subplot(gs[2,1])
    g2=nist_sum[nist_sum.benchmark=='NIST_MPC61_SemiEmpirical_Run1_to_Run2_20Folds']
    x_g2=np.arange(len(g2));v_g2=g2.j_life_mean.to_numpy();f_g2=g2.fail_pct_mean.to_numpy()
    ax_d.bar(x_g2,v_g2,.55,color=[P['gray'],P['purple'],P['orange'],P['navy'],P['red']])
    for i,(v,f) in enumerate(zip(v_g2,f_g2)):
        ax_d.text(i,v+.16,f'{v:.2f}\n({f:.1f}%)',ha='center',fontsize=7.1)
    ax_d.set_xticks(x_g2,['Cycle\nCP','Pure\nU-CP','Run1\nERM','Run2\nERM','DC-\nPdM'],fontsize=7.6)
    ax_d.set_ylim(0,7.35);ax_d.set_ylabel('Cost rate $\\hat{J}_{\\mathrm{life}}$')
    ax_d.set_title('(d) NIST 2.6.1.1.1 semi-emp.\n(50 trajs, unclipped)',loc='left',pad=8,fontsize=9.2)
    ax_d.set_axisbelow(True);ax_d.grid(axis='y',color='#DDE1E5',lw=.6)

    for d in folders:
        d.mkdir(parents=True,exist_ok=True)
        fig.savefig(d/'fig5_empirical_and_economic_regimes.pdf',bbox_inches='tight',pad_inches=0.06)
        fig.savefig(d/'fig5_empirical_and_economic_regimes.png',dpi=600,bbox_inches='tight',pad_inches=0.06)
    plt.close(fig)
    checks.append({'figure':'fig5_empirical_and_economic_regimes','pdf':'fig5_empirical_and_economic_regimes.pdf'})
    return checks,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parent.parent/'数据')
    ap.add_argument('--figures',type=Path,nargs='+',default=[Path(__file__).resolve().parent.parent/'图'])
    ap.add_argument('--suite',type=Path,default=Path(__file__).resolve().parent/'decision_calibrated_pdm_suite.py')
    args=ap.parse_args();checks,meta=draw(args.results,args.figures,args.suite)
    (args.figures[0]/'figure_build_manifest.json').write_text(json.dumps({'figures':checks,'sample':meta},indent=2),encoding='utf-8')
    print(json.dumps({'figures_created':len(checks),'sample':meta},indent=2))
if __name__=='__main__':main()
