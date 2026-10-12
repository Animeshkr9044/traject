from traject.calibrate import calibrate
from traject.diff import diff_runs
from traject.models import (
    Criterion,
    CriterionResult,
    ScoredTrajectory,
    Trajectory,
    Verdict,
)
from traject.report import diff_report_html, run_report_html

RUBRIC = [Criterion(id="a", desc="A"), Criterion(id="b", desc="B"), Criterion(id="c", desc="C")]


def _scored(sid: str, target: str, passes: dict[str, bool], task=True) -> ScoredTrajectory:
    traj = Trajectory(scenario_id=sid, target=target, turns=[], ended_by="max_turns")
    v = Verdict(
        criteria=[CriterionResult(id=k, passed=p, evidence="[0]", confidence=0.8) for k, p in passes.items()],
        task_completed=task, summary="",
    )
    return ScoredTrajectory(trajectory=traj, rubric=RUBRIC, verdict=v)


def test_diff_detects_regressions_improvements_and_winrate():
    a = [_scored("s1", "v1", {"a": True, "b": False, "c": True}),
         _scored("s2", "v1", {"a": True, "b": True, "c": True}),
         _scored("s3", "v1", {"a": False, "b": False, "c": False})]
    b = [_scored("s1", "v2", {"a": True, "b": True, "c": False}),   # b improved, c regressed, tie
         _scored("s2", "v2", {"a": True, "b": False, "c": True}),   # loss
         _scored("s4", "v2", {"a": True, "b": True, "c": True})]    # unmatched
    d = diff_runs(a, b)
    assert (d.wins, d.losses, d.ties) == (0, 1, 1)
    assert [(f.scenario_id, f.criterion_id) for f in d.regressions] == [("s1", "c"), ("s2", "b")]
    assert [(f.scenario_id, f.criterion_id) for f in d.improvements] == [("s1", "b")]
    assert d.only_in_a == ["s3"] and d.only_in_b == ["s4"]


def test_calibration_kappa_and_confusion():
    run = [_scored("s1", "v", {"a": True, "b": False, "c": True}),
           _scored("s2", "v", {"a": True, "b": True, "c": False})]
    labels = {("s1", "a"): True, ("s1", "b"): False, ("s1", "c"): False,  # judge too lenient on c
              ("s2", "a"): True, ("s2", "b"): False,                       # judge too lenient on b
              ("s9", "a"): True}                                           # unmatched
    rep = calibrate(run, labels)
    o = rep.overall
    assert o.n == 5 and o.agree == 3 and o.fp == 2 and o.fn == 0
    assert abs(o.agreement - 0.6) < 1e-9
    assert rep.unlabeled == 1 and rep.unmatched == 1
    assert rep.per_criterion["a"].kappa == 1.0 or rep.per_criterion["a"].agreement == 1.0
    assert len(rep.disagreements()) == 2


def test_kappa_perfect_and_chance():
    from traject.calibrate import PairStats
    assert PairStats(n=4, agree=4, tp=2, tn=2).kappa == 1.0
    # judge says yes to everything, human 50/50 -> kappa 0
    assert abs(PairStats(n=4, agree=2, tp=2, fp=2).kappa) < 1e-9


def test_reports_render_without_error():
    a = [_scored("s1", "v1", {"a": True, "b": False, "c": True})]
    b = [_scored("s1", "v2", {"a": True, "b": True, "c": True})]
    html = run_report_html(a)
    assert "<title>" in html and "s1" in html
    html2 = diff_report_html(diff_runs(a, b), a, b)
    assert "Regressions" in html2 and "v2" in html2
