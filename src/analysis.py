"""Deterministic in-memory primitives for the locked DGS analysis."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import hashlib, math, random
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

PRIMARY_METRICS=("M1","M2","M3")
PHASE0_MODELS=("google/gemma-2-2b-it","google/gemma-2-9b-it","Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-7B-Instruct","meta-llama/Llama-3.2-3B-Instruct")
PHASE0_SCREEN_TASKS=("DGS-034","DGS-026","DGS-010","DGS-003","DGS-018","DGS-037","DGS-030","DGS-014","DGS-005","DGS-022")
PHASE0_DIFFICULTIES={"DGS-003":"easy","DGS-010":"easy","DGS-018":"easy","DGS-026":"easy","DGS-034":"easy","DGS-005":"hard","DGS-014":"hard","DGS-022":"hard","DGS-030":"hard","DGS-037":"hard"}
_SIGN={"M1":-1.,"M2":1.,"M3":1.}
_TURNS=frozenset(("initial","measured","recovery","onset","onset_washout","irrelevant_control","irrelevant_control_washout","feedback_response_1","feedback_response_2","feedback_response_3","feedback_response_4","feedback_response_5"))
class AnalysisInputError(ValueError): pass
def _freeze(x): return MappingProxyType(dict(x))
def _text(x,n):
    if not isinstance(x,str) or not x: raise AnalysisInputError(n+" must be a nonempty string")
    return x
def _finite(x,n):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x): raise AnalysisInputError(n+" must be finite")
    return float(x)
def _count(x,n):
    if isinstance(x,bool) or not isinstance(x,int) or x<0: raise AnalysisInputError(n+" must be a nonnegative integer")
    return x

@dataclass(frozen=True)
class AnalysisObservation:
    experiment_phase:str; run_id:str; split:str; model_id:str; task_id:str; cell_id:str; difficulty:str; feedback_validity:str; tone:str; turn:str; metric_name:str; metric_value:float|None; missing_reason:str|None; correctness:bool|None; generated_response_tokens:int; false_negative_history_eligible:bool; analysis_feedback_validity:str|None=None
    def __post_init__(self):
        for n in ("experiment_phase","run_id","split","model_id","task_id","cell_id","difficulty","feedback_validity","tone","turn","metric_name"): _text(getattr(self,n),n)
        if self.experiment_phase not in ("phase_0","phase_1") or self.split not in ("discovery","holdout"): raise AnalysisInputError("unsupported phase or split")
        if self.difficulty not in ("easy","hard") or self.tone not in ("neutral","hostile") or self.feedback_validity not in ("accurate","malfunctioning_always_fail") or self.turn not in _TURNS: raise AnalysisInputError("unsupported analysis factors")
        if self.metric_name not in PRIMARY_METRICS or self.cell_id.split("__") != [self.difficulty,self.feedback_validity,self.tone]: raise AnalysisInputError("invalid metric or contradictory cell factors")
        if (self.metric_value is None)==(self.missing_reason is None): raise AnalysisInputError("exactly one metric value or missing reason")
        if self.metric_value is not None: object.__setattr__(self,"metric_value",_finite(self.metric_value,"metric_value"))
        elif not isinstance(self.missing_reason,str) or not self.missing_reason: raise AnalysisInputError("missing reason required")
        if self.correctness is not None and not isinstance(self.correctness,bool): raise AnalysisInputError("correctness must be bool or None")
        if not isinstance(self.false_negative_history_eligible,bool): raise AnalysisInputError("eligibility must be bool")
        _count(self.generated_response_tokens,"generated_response_tokens")
        if self.analysis_feedback_validity is not None and self.analysis_feedback_validity not in ("accurate","malfunctioning_always_fail"): raise AnalysisInputError("invalid analysis label")
    @property
    def effective_feedback_validity(self): return self.analysis_feedback_validity or self.feedback_validity
    @property
    def key(self): return (self.experiment_phase,self.run_id,self.split,self.model_id,self.task_id,self.cell_id,self.turn,self.metric_name)

def validate_observations(rows:Iterable[AnalysisObservation],*,experiment_phase=None,split=None,measured_only=False):
    rows=tuple(rows)
    if any(not isinstance(r,AnalysisObservation) for r in rows) or len({r.key for r in rows})!=len(rows): raise AnalysisInputError("invalid or duplicate observation")
    seen={}
    for r in rows:
        if r.task_id in seen and seen[r.task_id]!=r.difficulty: raise AnalysisInputError("task has contradictory difficulty")
        seen[r.task_id]=r.difficulty
    if experiment_phase and any(r.experiment_phase!=experiment_phase for r in rows): raise AnalysisInputError("mixed experiment phases")
    if split and any(r.split!=split for r in rows): raise AnalysisInputError("mixed splits")
    if measured_only and any(r.turn!="measured" for r in rows): raise AnalysisInputError("measured rows required")
    return rows
@dataclass(frozen=True)
class Standardization:
    mean:float|None; sample_sd:float|None; unavailable_reason:str|None=None
    @property
    def available(self): return self.unavailable_reason is None
def freeze_neutral_standardization(rows):
    rows=validate_observations(rows); keys={(r.model_id,r.metric_name) for r in rows}; groups={}
    for r in rows:
        if r.split=="discovery" and r.turn=="measured" and r.feedback_validity=="accurate" and r.tone=="neutral" and r.metric_value is not None: groups.setdefault((r.model_id,r.metric_name),[]).append(r.metric_value)
    out={}
    for k in keys:
        v=sorted(groups.get(k,[]))
        if len(v)<2: out[k]=Standardization(None,None,"insufficient_neutral_observations")
        else:
            m=mean(v); sd=math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)); out[k]=Standardization(m,sd) if sd else Standardization(None,None,"zero_neutral_sample_sd")
    return _freeze(out)
def standardized_value(r,frozen):
    s=frozen.get((r.model_id,r.metric_name))
    return None if r.metric_value is None or s is None or not s.available else (r.metric_value-s.mean)/s.sample_sd
def benjamini_hochberg(values):
    mapping=isinstance(values,Mapping); pairs=list(values.items()) if mapping else list(enumerate(values))
    for _,p in pairs:
        _finite(p,"p value")
        if not 0<=p<=1: raise AnalysisInputError("p values must be in [0,1]")
    ordered=sorted(enumerate(pairs),key=lambda x:(x[1][1],x[0])); result=[0.]*len(pairs); running=1.
    for rank,(original,(_,p)) in reversed(list(enumerate(ordered))): running=min(running,min(1.,p*len(pairs)/(rank+1))); result[original]=running
    return _freeze({k:result[i] for i,(k,_) in enumerate(pairs)}) if mapping else tuple(result)

@dataclass(frozen=True)
class MetricScreen: signed_delta:float|None; unavailable_reason:str|None=None
@dataclass(frozen=True)
class ModelScreen: model_id:str; metrics:Mapping[str,MetricScreen]; score:float|None; coherent:bool
@dataclass(frozen=True)
class Phase0Selection: status:str; models:Mapping[str,ModelScreen]; primary_model_id:str|None=None; control_model_id:str|None=None; blocked_reason:str|None=None
def phase0_screen(rows,configured_models=PHASE0_MODELS,screen_task_ids=PHASE0_SCREEN_TASKS):
    rows=tuple(rows)
    if tuple(configured_models)!=PHASE0_MODELS or tuple(screen_task_ids)!=PHASE0_SCREEN_TASKS: return Phase0Selection("blocked",_freeze({}),blocked_reason="frozen_phase0_configuration_mismatch")
    try: validate_observations(rows,experiment_phase="phase_0",split="discovery",measured_only=True)
    except AnalysisInputError as e: return Phase0Selection("blocked",_freeze({}),blocked_reason="invalid_phase0_rows:"+str(e))
    expected={(m,t,v,"neutral",x) for m in PHASE0_MODELS for t in PHASE0_SCREEN_TASKS for v in ("accurate","malfunctioning_always_fail") for x in PRIMARY_METRICS}
    if len({r.run_id for r in rows})!=1: return Phase0Selection("blocked",_freeze({}),blocked_reason="phase0_requires_exactly_one_run")
    if any(r.difficulty != PHASE0_DIFFICULTIES.get(r.task_id) for r in rows): return Phase0Selection("blocked",_freeze({}),blocked_reason="phase0_task_difficulty_mismatch")
    got={(r.model_id,r.task_id,r.feedback_validity,r.tone,r.metric_name) for r in rows}
    if got!=expected or len(rows)!=len(expected): return Phase0Selection("blocked",_freeze({}),blocked_reason="incomplete_or_extra_phase0_screen_rows")
    by={(r.model_id,r.task_id,r.feedback_validity,r.metric_name):r for r in rows}; frozen=freeze_neutral_standardization(rows); out={}
    for m in PHASE0_MODELS:
        metrics={}
        for x in PRIMARY_METRICS:
            s=frozen[(m,x)]; pairs=[(by[(m,t,"accurate",x)],by[(m,t,"malfunctioning_always_fail",x)]) for t in PHASE0_SCREEN_TASKS]
            if not s.available: metrics[x]=MetricScreen(None,"neutral_standardization_unavailable")
            elif any(a.metric_value is None or b.metric_value is None for a,b in pairs): metrics[x]=MetricScreen(None,"screen_endpoint_qc_unavailable")
            else: metrics[x]=MetricScreen(_SIGN[x]*mean(b.metric_value-a.metric_value for a,b in pairs)/s.sample_sd)
        avail=[v.signed_delta for v in metrics.values() if v.signed_delta is not None]; score=mean(avail) if avail else None; out[m]=ModelScreen(m,_freeze(metrics),score,bool(score is not None and score>0 and sum(x>0 for x in avail)>=2))
    coherent=[out[m] for m in PHASE0_MODELS if out[m].coherent]
    if not any(item.score is not None for item in out.values()):return Phase0Selection("blocked",_freeze(out),blocked_reason="all_phase0_metrics_unavailable")
    if not coherent:return Phase0Selection("escalation_required",_freeze(out),blocked_reason="all_model_screen_null")
    primary=sorted(coherent,key=lambda a:(-a.score,-(a.metrics["M1"].signed_delta is not None),-(a.metrics["M1"].signed_delta if a.metrics["M1"].signed_delta is not None else float("-inf")),a.model_id))[0]
    q=[x for x in out.values() if x.model_id.startswith("Qwen/") and x.score is not None and x.model_id!=primary.model_id]
    if not q:return Phase0Selection("blocked",_freeze(out),primary.model_id,blocked_reason="no_distinct_available_qwen_control")
    control=min(q,key=lambda x:(abs(x.score),PHASE0_MODELS.index(x.model_id)))
    return Phase0Selection("selected",_freeze(out),primary.model_id,control.model_id)

@dataclass(frozen=True)
class CoefficientResult:
    coefficient:float; standard_error:float; ci95:tuple[float,float]; raw_p:float; adjusted_p:float|None; sign_aligned_coefficient:float; qualifying:bool; instability_positive:bool
    @property
    def instability_positive_qualifying(self): return self.qualifying and self.instability_positive
@dataclass(frozen=True)
class PairedDescriptor:
    contrast:str;n_pairs:int;n_items:int;raw_mean:float|None;sign_aligned_mean:float|None;raw_ci95:tuple[float,float]|None;sign_aligned_ci95:tuple[float,float]|None;unavailable_reason:str|None=None
@dataclass(frozen=True)
class G1MetricResult:
    model_id:str; metric_name:str; validity:CoefficientResult|None; tone:CoefficientResult|None; n_rows:int; n_items:int; converged:bool; unavailable_reason:str|None=None; paired_validity:PairedDescriptor|None=None; paired_tone:PairedDescriptor|None=None
def _paired_descriptor(rows, frozen, metric, contrast):
    """Raw factorial paired descriptor; bootstrap seed DGS-AC1-G1-PAIRED-v1."""
    if any(r.analysis_feedback_validity is not None for r in rows):
        return PairedDescriptor(
            contrast, 0, 0, None, None, None, None,
            "unavailable_for_shuffled_analysis_labels",
        )
    standardization = frozen.get((rows[0].model_id, metric)) if rows else None
    if standardization is None or not standardization.available:
        return PairedDescriptor(
            contrast, 0, 0, None, None, None, None,
            "neutral_standardization_unavailable",
        )
    groups = {}
    for r in rows:
        z = standardized_value(r, frozen)
        if z is None:
            continue
        if contrast == "validity":
            key = (r.task_id, r.tone)
            label = r.feedback_validity
            left, right = "malfunctioning_always_fail", "accurate"
        else:
            key = (r.task_id, r.feedback_validity)
            label = r.tone
            left, right = "hostile", "neutral"
        groups.setdefault(key, {})[label] = z
    pairs = [(key[0], values[left] - values[right]) for key, values in groups.items() if left in values and right in values]
    by_item = {}
    for item, value in pairs:
        by_item.setdefault(item, []).append(value)
    if len(by_item) < 2:
        return PairedDescriptor(
            contrast, len(pairs), len(by_item), None, None, None, None,
            "at_least_two_items_required",
        )
    seed_text = "DGS-AC1-G1-PAIRED-v1|%s|%s|%s" % (rows[0].model_id, metric, contrast)
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    items = sorted(by_item)
    bootstrap = []
    for _ in range(2000):
        sampled_items = [rng.choice(items) for _ in items]
        bootstrap.append(mean(value for item in sampled_items for value in by_item[item]))
    bootstrap.sort()

    def quantile(probability):
        position = (len(bootstrap) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return bootstrap[lower] + (bootstrap[upper] - bootstrap[lower]) * (position - lower)

    raw = mean(value for _, value in pairs)
    raw_ci95 = (quantile(.025), quantile(.975))
    sign_aligned_ci95 = tuple(sorted(_SIGN[metric] * value for value in raw_ci95))
    return PairedDescriptor(
        contrast,
        len(pairs),
        len(by_item),
        raw,
        _SIGN[metric] * raw,
        raw_ci95,
        sign_aligned_ci95,
    )
def g1_adjusted_effects(rows,metrics=PRIMARY_METRICS):
    rows = validate_observations(rows, experiment_phase="phase_1", split="discovery", measured_only=True)
    metrics = tuple(metrics)
    if not metrics or len(metrics) != len(set(metrics)) or any(metric not in PRIMARY_METRICS for metric in metrics):
        raise AnalysisInputError("invalid metric family")
    if len({row.run_id for row in rows}) != 1:
        raise AnalysisInputError("G1 requires exactly one phase_1 discovery run")
    frozen = freeze_neutral_standardization(rows)
    out = {}
    for model in sorted({r.model_id for r in rows}):
        for metric in metrics:
            source = sorted(
                (row for row in rows if row.model_id == model and row.metric_name == metric),
                key=lambda row: row.key,
            )
            paired_validity = _paired_descriptor(source, frozen, metric, "validity")
            paired_tone = _paired_descriptor(source, frozen, metric, "tone")
            complete = [
                row for row in source
                if row.metric_value is not None
                and row.correctness is not None
                and standardized_value(row, frozen) is not None
            ]
            standardization = frozen.get((model, metric), Standardization(None, None, "missing"))
            bad = not standardization.available
            covariates_present = (
                {row.effective_feedback_validity for row in complete} >= {"accurate", "malfunctioning_always_fail"}
                and {row.tone for row in complete} >= {"neutral", "hostile"}
                and {row.difficulty for row in complete} >= {"easy", "hard"}
                and len({row.correctness for row in complete}) > 1
                and len({row.generated_response_tokens for row in complete}) > 1
                and len({row.task_id for row in complete}) > 1
            )
            result_fields = (model, metric, None, None, len(complete), len({row.task_id for row in complete}), False)
            if bad or not covariates_present:
                out[(model, metric)] = G1MetricResult(
                    *result_fields,
                    "neutral_standardization_unavailable" if bad else "required_classes_or_covariates_absent",
                    paired_validity,
                    paired_tone,
                )
                continue
            try:
                import pandas as pd
                from statsmodels.regression.mixed_linear_model import MixedLM

                frame = pd.DataFrame({
                    "z_metric": [standardized_value(row, frozen) for row in complete],
                    "malfunctioning": [int(row.effective_feedback_validity == "malfunctioning_always_fail") for row in complete],
                    "hostile": [int(row.tone == "hostile") for row in complete],
                    "difficulty_hard": [int(row.difficulty == "hard") for row in complete],
                    "correctness": [int(row.correctness) for row in complete],
                    "length": [row.generated_response_tokens for row in complete],
                    "item": [row.task_id for row in complete],
                })
                frame["length"] = (frame["length"] - frame["length"].mean()) / frame["length"].std(ddof=0)
                fit = MixedLM.from_formula(
                    "z_metric ~ malfunctioning + hostile + difficulty_hard + correctness + length",
                    groups="item",
                    data=frame,
                ).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
                if not fit.converged:
                    raise RuntimeError()

                def coefficient(name):
                    value = _finite(fit.params[name], name)
                    error = _finite(fit.bse[name], name)
                    probability = _finite(fit.pvalues[name], name)
                    return CoefficientResult(
                        value,
                        error,
                        (value - 1.95996398454 * error, value + 1.95996398454 * error),
                        probability,
                        None,
                        _SIGN[metric] * value,
                        False,
                        _SIGN[metric] * value > 0,
                    )

                out[(model, metric)] = G1MetricResult(
                    model,
                    metric,
                    coefficient("malfunctioning"),
                    coefficient("hostile"),
                    len(complete),
                    len({row.task_id for row in complete}),
                    True,
                    paired_validity=paired_validity,
                    paired_tone=paired_tone,
                )
            except Exception as error:
                out[(model, metric)] = G1MetricResult(
                    *result_fields,
                    "mixedlm_unavailable:" + type(error).__name__,
                    paired_validity,
                    paired_tone,
                )
    family={(m,x,k):getattr(r,k).raw_p for (m,x),r in out.items() if r.validity and r.tone for k in ("validity","tone")}; adj=benjamini_hochberg(family) if family else {}
    for (m,x),r in list(out.items()):
      if r.validity:
        upd=lambda c,k:replace(c,adjusted_p=adj[(m,x,k)],qualifying=adj[(m,x,k)]<.01)
        out[(m,x)]=replace(r,validity=upd(r.validity,"validity"),tone=upd(r.tone,"tone"))
    return _freeze(out)

def shuffled_feedback_labels(rows):
    rows=validate_observations(rows); logical={}; strata={}
    if len({r.experiment_phase for r in rows})!=1 or len({r.run_id for r in rows})!=1 or len({r.split for r in rows})!=1:
      raise AnalysisInputError("shuffle requires one phase, run, and split")
    for r in rows:
      key=(r.experiment_phase,r.run_id,r.model_id,r.task_id,r.cell_id)
      if key in logical and (logical[key].feedback_validity,logical[key].difficulty,logical[key].tone)!=(r.feedback_validity,r.difficulty,r.tone):raise AnalysisInputError("contradictory logical item-cell factors")
      logical[key]=r
    for key,r in logical.items():strata.setdefault((r.experiment_phase,r.run_id,r.model_id,r.difficulty,r.tone),[]).append((key,r))
    labels={}
    for (_,_,m,_,_),g in strata.items():
      rank=sorted(g,key=lambda x:hashlib.sha256(("DGS-AC1-SHUFFLE-v1|%s|%s|%s"%(m,x[1].task_id,x[1].cell_id)).encode()).hexdigest());n=sum(r.feedback_validity=="malfunctioning_always_fail" for _,r in g)
      for i,(key,_) in enumerate(rank):labels[key]="malfunctioning_always_fail" if i<n else "accurate"
    return tuple(sorted((replace(r,analysis_feedback_validity=labels[(r.experiment_phase,r.run_id,r.model_id,r.task_id,r.cell_id)]) for r in rows),key=lambda r:r.key))

@dataclass(frozen=True)
class ReversalRow:
    experiment_phase:str;run_id:str;split:str;model_id:str;metric_name:str;task_id:str;tone:str;measured_accurate:float|None;measured_malfunctioning:float|None;post_correction_malfunctioning:float|None;false_negative_history_eligible:bool
    def __post_init__(self):
      if self.experiment_phase!="phase_1" or self.split!="discovery":raise AnalysisInputError("G2 requires phase_1 discovery rows")
      for name in ("run_id","model_id","task_id"): _text(getattr(self,name),name)
      if self.metric_name not in PRIMARY_METRICS or self.tone not in ("neutral","hostile") or not isinstance(self.false_negative_history_eligible,bool):raise AnalysisInputError("invalid reversal row")
      for n in ("measured_accurate","measured_malfunctioning","post_correction_malfunctioning"):
       if getattr(self,n) is not None:object.__setattr__(self,n,_finite(getattr(self,n),n))
@dataclass(frozen=True)
class G2Result:
    model_id:str|None;metric_name:str|None;n_items:int;n_rows:int;induction:float|None;recovery:float|None;recovery_to_induction:float|None;recovery_ci95:tuple[float,float]|None;unavailable_reason:str|None=None
def g2_reversal(rows,*,bootstrap_samples=2000):
    rows=tuple(rows)
    if bootstrap_samples!=2000:raise AnalysisInputError("G2 bootstrap count is frozen at 2000")
    if not rows or any(not isinstance(r,ReversalRow) for r in rows) or len({r.run_id for r in rows})!=1 or len({r.model_id for r in rows})!=1 or len({r.metric_name for r in rows})!=1 or len({(r.task_id,r.tone) for r in rows})!=len(rows):raise AnalysisInputError("invalid G2 rows")
    m,x=rows[0].model_id,rows[0].metric_name; good=[r for r in rows if r.false_negative_history_eligible]
    if not good:return G2Result(m,x,0,0,None,None,None,None,"no_false_negative_eligible_rows")
    if any(None in (r.measured_accurate,r.measured_malfunctioning,r.post_correction_malfunctioning) for r in good):return G2Result(m,x,len({r.task_id for r in good}),len(good),None,None,None,None,"required_reversal_endpoint_missing")
    by={}
    for r in good:by.setdefault(r.task_id,[]).append((_SIGN[x]*(r.measured_malfunctioning-r.measured_accurate),_SIGN[x]*(r.measured_malfunctioning-r.post_correction_malfunctioning)))
    if len(by)<2:return G2Result(m,x,len(by),len(good),None,None,None,None,"at_least_two_items_required_for_cluster_ci")
    p=[v for vals in by.values() for v in vals]; ind,rec=mean(v[0] for v in p),mean(v[1] for v in p);items=sorted(by);rng=random.Random(int.from_bytes(hashlib.sha256(("DGS-AC1-G2-BOOTSTRAP-v1|%s|%s"%(m,x)).encode()).digest()[:8],"big")); bs=[mean(v[1] for i in [rng.choice(items) for _ in items] for v in by[i]) for _ in range(2000)];bs.sort()
    def q(z):
      p=(len(bs)-1)*z;a,b=math.floor(p),math.ceil(p);return bs[a]+(bs[b]-bs[a])*(p-a)
    return G2Result(m,x,len(by),len(good),ind,rec,rec/ind if ind else None,(q(.025),q(.975)))

@dataclass(frozen=True)
class G5Row:
    experiment_phase:str;run_id:str;split:str;model_id:str;task_id:str;cell_id:str;difficulty:str;feedback_validity:str;tone:str;turn:str;metrics:Mapping[str,float|None];correctness:bool|None;generated_response_tokens:int;analysis_feedback_validity:str|None=None
    def __post_init__(self):
      for name in ("run_id","model_id","task_id","cell_id"): _text(getattr(self,name),name)
      if self.experiment_phase!="phase_1" or self.split!="discovery" or self.turn!="measured" or self.cell_id.split("__")!=[self.difficulty,self.feedback_validity,self.tone]:raise AnalysisInputError("invalid G5 row")
      if self.difficulty not in ("easy","hard") or self.tone not in ("neutral","hostile") or self.feedback_validity not in ("accurate","malfunctioning_always_fail") or self.correctness is not None and not isinstance(self.correctness,bool):raise AnalysisInputError("invalid G5 factors")
      if self.analysis_feedback_validity is not None and self.analysis_feedback_validity not in ("accurate","malfunctioning_always_fail"):raise AnalysisInputError("invalid G5 analysis label")
      if set(self.metrics)-set(PRIMARY_METRICS):raise AnalysisInputError("invalid G5 metric keys")
      _count(self.generated_response_tokens,"length");object.__setattr__(self,"metrics",_freeze({k:None if v is None else _finite(v,k) for k,v in self.metrics.items()}))
    @property
    def effective_feedback_validity(self):return self.analysis_feedback_validity or self.feedback_validity
def _validate_g5_factorial(rows):
    """Validate raw, pre-QC factorial source cells for G5 and its shuffle null."""
    if not rows or any(not isinstance(row, G5Row) for row in rows):
        raise AnalysisInputError("invalid G5 factorial source")
    if len({row.run_id for row in rows}) != 1:
        raise AnalysisInputError("G5 source requires exactly one run")
    if len({row.model_id for row in rows}) != 1:
        raise AnalysisInputError("G5 source requires exactly one model")

    by_task = {}
    for row in rows:
        by_task.setdefault(row.task_id, []).append(row)
    difficulties = set()
    for group in by_task.values():
        group_difficulties = {row.difficulty for row in group}
        if len(group_difficulties) != 1:
            raise AnalysisInputError("task difficulty mismatch")
        difficulty = next(iter(group_difficulties))
        difficulties.add(difficulty)
        expected = {
            f"{difficulty}__{validity}__{tone}"
            for validity in ("accurate", "malfunctioning_always_fail")
            for tone in ("neutral", "hostile")
        }
        if {row.cell_id for row in group} != expected or len(group) != 4:
            raise AnalysisInputError("missing or extra factorial source cell")
    if difficulties != {"easy", "hard"}:
        raise AnalysisInputError("both difficulty strata required")
def g5_shuffled_feedback_labels(rows):
    rows=tuple(rows)
    if not rows or any(not isinstance(r,G5Row) for r in rows) or len({r.run_id for r in rows})!=1 or len({r.model_id for r in rows})!=1 or len({(r.run_id,r.model_id,r.task_id,r.cell_id) for r in rows})!=len(rows):raise AnalysisInputError("invalid G5 shuffle dataset")
    _validate_g5_factorial(rows)
    labels={};groups={}
    for r in rows:groups.setdefault((r.model_id,r.difficulty,r.tone),[]).append(r)
    for (m,_,_),g in groups.items():
      rank=sorted(g,key=lambda r:hashlib.sha256(("DGS-AC1-SHUFFLE-v1|%s|%s|%s"%(m,r.task_id,r.cell_id)).encode()).hexdigest());n=sum(r.feedback_validity=="malfunctioning_always_fail" for r in g)
      for i,r in enumerate(rank):labels[(r.run_id,r.model_id,r.task_id,r.cell_id)]="malfunctioning_always_fail" if i<n else "accurate"
    return tuple(sorted((replace(r,analysis_feedback_validity=labels[(r.run_id,r.model_id,r.task_id,r.cell_id)]) for r in rows),key=lambda r:(r.model_id,r.task_id,r.cell_id)))
@dataclass(frozen=True)
class G5Result:
    full_auc:float|None;baseline_auc:float|None;auc_gap:float|None;n_rows:int;dropped_count:int;n_folds:int;fold_item_ids:tuple[tuple[str,...],...];heldout_probabilities:Mapping[tuple[str,str],tuple[float,float]]=field(default_factory=lambda:_freeze({}));unavailable_reason:str|None=None
    @property
    def gap_at_least_point_one(self):return self.auc_gap is not None and self.auc_gap>=.1
def g5_predictive_gap(rows,eligible_metrics):
    rows = tuple(rows)
    metrics = tuple(eligible_metrics)
    if (
        not rows
        or any(not isinstance(row, G5Row) for row in rows)
        or len({row.run_id for row in rows}) != 1
        or len({row.model_id for row in rows}) != 1
        or len({(row.task_id, row.cell_id) for row in rows}) != len(rows)
    ):
        raise AnalysisInputError("invalid G5 dataset")
    if (
        not metrics
        or len(metrics) != len(set(metrics))
        or any(metric not in PRIMARY_METRICS for metric in metrics)
    ):
        raise AnalysisInputError("invalid G5 metrics")
    try:
        _validate_g5_factorial(rows)
    except AnalysisInputError as error:
        return G5Result(
            None,
            None,
            None,
            0,
            0,
            0,
            (),
            unavailable_reason="invalid_factorial_source:" + str(error),
        )
    kept = [
        row for row in rows
        if row.correctness is not None and all(row.metrics.get(metric) is not None for metric in metrics)
    ]
    dropped_count = len(rows) - len(kept)
    if not kept:
        return G5Result(None, None, None, 0, dropped_count, 0, (), unavailable_reason="no_complete_case_rows")
    items = sorted({row.task_id for row in kept})
    if len(items) < 2:
        return G5Result(None, None, None, len(kept), dropped_count, 0, (), unavailable_reason="at_least_two_items_required")
    folds = tuple((item,) for item in items)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="sklearn_unavailable")
    labels = []
    full_probabilities = []
    baseline_probabilities = []
    heldout = {}
    for fold in folds:
        test = sorted((row for row in kept if row.task_id in fold), key=lambda row: (row.task_id, row.cell_id))
        train = [row for row in kept if row.task_id not in fold]
        classes = {
            label: sorted(
                (row for row in train if int(row.effective_feedback_validity == "malfunctioning_always_fail") == label),
                key=lambda row: (row.task_id, row.cell_id),
            )
            for label in (0, 1)
        }
        if not classes[0] or not classes[1]:
            return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="one_class_training_fold")
        balanced_count = min(map(len, classes.values()))
        train = sorted(classes[0][:balanced_count] + classes[1][:balanced_count], key=lambda row: (row.task_id, row.cell_id))

        def features(source, full):
            if full:
                return [[float(row.metrics[metric]) for metric in metrics] for row in source]
            return [[float(row.correctness), float(row.generated_response_tokens)] for row in source]

        def scale(train_values, test_values):
            columns = list(zip(*train_values))
            means = [mean(column) for column in columns]
            deviations = [math.sqrt(mean((value - average) ** 2 for value in column)) for column, average in zip(columns, means)]

            def standardized(values):
                return [
                    [0.0 if deviation == 0 else (value - average) / deviation for value, average, deviation in zip(row, means, deviations)]
                    for row in values
                ]

            return standardized(train_values), standardized(test_values)

        full_train, full_test = scale(features(train, True), features(test, True))
        baseline_train, baseline_test = scale(features(train, False), features(test, False))
        train_labels = [int(row.effective_feedback_validity == "malfunctioning_always_fail") for row in train]
        try:
            full_model = LogisticRegression(
            C=1,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=0,
            ).fit(full_train, train_labels)
            baseline_model = LogisticRegression(
            C=1,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=0,
            ).fit(baseline_train, train_labels)
        except Exception as error:
            return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="logistic_fit_failed:" + type(error).__name__)
        if any(iterations >= 1000 for iterations in full_model.n_iter_) or any(iterations >= 1000 for iterations in baseline_model.n_iter_):
            return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="logistic_nonconvergence")
        fold_full = full_model.predict_proba(full_test)[:, 1].tolist()
        fold_baseline = baseline_model.predict_proba(baseline_test)[:, 1].tolist()
        if not all(math.isfinite(value) for value in fold_full + fold_baseline):
            return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="nonfinite_probabilities")
        for row, full_probability, baseline_probability in zip(test, fold_full, fold_baseline):
            heldout[(row.task_id, row.cell_id)] = (float(full_probability), float(baseline_probability))
            labels.append(int(row.effective_feedback_validity == "malfunctioning_always_fail"))
            full_probabilities.append(full_probability)
            baseline_probabilities.append(baseline_probability)
    if len(set(labels)) < 2:
        return G5Result(None, None, None, len(kept), dropped_count, 0, folds, unavailable_reason="one_class_oof")
    full_auc = float(roc_auc_score(labels, full_probabilities))
    baseline_auc = float(roc_auc_score(labels, baseline_probabilities))
    return G5Result(full_auc, baseline_auc, full_auc - baseline_auc, len(kept), dropped_count, len(folds), folds, _freeze(heldout))
