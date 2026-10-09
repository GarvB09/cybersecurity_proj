import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

rng = np.random.default_rng(42)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 11})
BLUE, NAVY, LIGHT = "#1f5aa6", "#0b2e59", "#9cc0e6"

FEATURES = ["auth_fail_ratio", "req_rate", "uniq_endpoints", "off_hours", "new_asn",
            "session_reuse_ips", "admin_api_ratio", "bytes_out_mb", "integrity_drift",
            "outbound_conns", "new_processes", "config_changes"]

# Each class is a profile of window-level telemetry (5-min windows per appliance/source).
PROFILES = {
    "benign":      dict(auth_fail_ratio=(0.03, .03), req_rate=(12, 6), uniq_endpoints=(6, 3), off_hours=.15, new_asn=.03,
                        session_reuse_ips=(1, .2), admin_api_ratio=(.01, .01), bytes_out_mb=(2, 1.5), integrity_drift=.002,
                        outbound_conns=(1, 1), new_processes=(.2, .4), config_changes=(.05, .2)),
    "recon":       dict(auth_fail_ratio=(.25, .1), req_rate=(60, 20), uniq_endpoints=(35, 10), off_hours=.5, new_asn=.8,
                        session_reuse_ips=(1, .2), admin_api_ratio=(.3, .1), bytes_out_mb=(1, .8), integrity_drift=.0,
                        outbound_conns=(1, 1), new_processes=(.2, .4), config_changes=(0, .1)),
    "exploitation": dict(auth_fail_ratio=(.05, .05), req_rate=(20, 8), uniq_endpoints=(8, 3), off_hours=.6, new_asn=.7,
                        session_reuse_ips=(1, .3), admin_api_ratio=(.6, .15), bytes_out_mb=(3, 2), integrity_drift=.4,
                        outbound_conns=(5, 3), new_processes=(4, 2), config_changes=(.5, .6)),
    "persistence": dict(auth_fail_ratio=(.02, .02), req_rate=(5, 3), uniq_endpoints=(3, 1.5), off_hours=.7, new_asn=.6,
                        session_reuse_ips=(1, .2), admin_api_ratio=(.2, .1), bytes_out_mb=(4, 3), integrity_drift=.85,
                        outbound_conns=(3, 2), new_processes=(2, 1.5), config_changes=(1.5, 1)),
    "cred_session": dict(auth_fail_ratio=(.04, .03), req_rate=(15, 6), uniq_endpoints=(9, 4), off_hours=.55, new_asn=.65,
                        session_reuse_ips=(3.5, 1), admin_api_ratio=(.08, .05), bytes_out_mb=(25, 12), integrity_drift=.1,
                        outbound_conns=(6, 3), new_processes=(.5, .6), config_changes=(.1, .3)),
}
# A "novel" variant: different entry point, quieter footprint — stands in for a mitigation-bypass zero-day.
PROFILES["novel_variant"] = dict(PROFILES["exploitation"], admin_api_ratio=(.15, .08), req_rate=(9, 4),
                                 new_processes=(2.5, 1.5), integrity_drift=.3, off_hours=.4)


# Legitimate admin maintenance: off-hours, config edits, process spawns, integrity drift after upgrades.
PROFILES["benign_admin"] = dict(PROFILES["benign"], off_hours=.8, admin_api_ratio=(.35, .15), config_changes=(1.2, .9),
                                new_processes=(2, 1.5), integrity_drift=.15, outbound_conns=(3, 2))


def raw(cls, n):
    p, cols = PROFILES[cls], {}
    for f in FEATURES:
        v = p[f]
        cols[f] = rng.binomial(1, v, n).astype(float) if isinstance(v, float) else np.clip(rng.normal(v[0], v[1], n), 0, None)
    return pd.DataFrame(cols)


def sample(cls, n):
    df = raw(cls, n)
    if cls not in ("benign", "benign_admin"):
        # Stealthy windows: blend part of each attack window toward normal behaviour.
        b = rng.uniform(0, 0.75, (n, 1))
        df = pd.DataFrame(b * raw("benign", n).values + (1 - b) * df.values, columns=FEATURES)
        for f in ["off_hours", "new_asn", "integrity_drift"]:
            df[f] = (df[f] >= 0.5).astype(float)
    df["stage"] = cls
    df["label"] = int(cls not in ("benign", "benign_admin"))
    return df


def rules(df):
    hit = (df.integrity_drift == 1) | ((df.new_processes > 3) & (df.outbound_conns > 4)) | (df.session_reuse_ips >= 3)
    return hit.astype(int).values


data = pd.concat([sample("benign", 19000), sample("benign_admin", 1000), sample("recon", 600), sample("exploitation", 600),
                  sample("persistence", 400), sample("cred_session", 400)], ignore_index=True)
novel = pd.concat([sample("benign", 3800), sample("benign_admin", 200), sample("novel_variant", 300)], ignore_index=True)

train, test = train_test_split(data, test_size=0.3, stratify=data.stage, random_state=7)
X_tr, X_te = train[FEATURES].values, test[FEATURES].values

iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=7).fit(train[train.label == 0][FEATURES].values)
rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=7).fit(X_tr, train.label)

iso_thr = np.quantile(-iso.score_samples(train[train.label == 0][FEATURES].values), 0.99)


def scores(df):
    X = df[FEATURES].values
    s_iso = -iso.score_samples(X)
    s_rf = rf.predict_proba(X)[:, 1]
    risk = np.clip(0.7 * s_rf + 0.15 * (s_iso > iso_thr) + 0.15 * rules(df), 0, 1)
    return dict(rules=rules(df), iso=(s_iso > iso_thr).astype(int), rf=(s_rf > .5).astype(int),
                hybrid=(risk >= 0.40).astype(int)), dict(iso=s_iso, rf=s_rf, hybrid=risk)


def metrics(y, preds, probs):
    out = {}
    for k, p in preds.items():
        tn, fp, fn, tp = confusion_matrix(y, p).ravel()
        out[k] = dict(precision=round(precision_score(y, p), 3), recall=round(recall_score(y, p), 3),
                      f1=round(f1_score(y, p), 3), fpr=round(fp / (fp + tn), 4),
                      auc=round(roc_auc_score(y, probs[k]), 3) if k in probs else None)
    return out


p_te, s_te = scores(test)
p_nv, s_nv = scores(novel)
res = dict(test=metrics(test.label, p_te, s_te), novel=metrics(novel.label, p_nv, s_nv))
res["stage_recall_hybrid"] = {st: round(p_te["hybrid"][(test.stage == st).values].mean(), 3)
                              for st in ["recon", "exploitation", "persistence", "cred_session"]}
res["counts"] = dict(train=len(train), test=len(test), novel=len(novel))
res["confusion_hybrid"] = confusion_matrix(test.label, p_te["hybrid"]).tolist()

# Automated response simulation: time from first malicious window to containment.
PLAYBOOK = [("Alert triage and enrichment", 2, 45), ("Block source IP / ASN at edge", 1, 30),
            ("Revoke active VPN sessions", 1, 40), ("Isolate appliance to mgmt VLAN", 3, 90),
            ("Force credential reset for affected users", 5, 240), ("Snapshot + external integrity scan", 10, 180)]
auto = np.cumsum([a for _, a, _ in PLAYBOOK]); manual = np.cumsum([m for *_, m in PLAYBOOK])
res["response"] = dict(steps=[s for s, *_ in PLAYBOOK], auto_min=auto.tolist(), manual_min=manual.tolist())
res["feature_importance"] = dict(sorted(zip(FEATURES, rf.feature_importances_.round(4)), key=lambda x: -x[1]))
json.dump(res, open("results.json", "w"), indent=2, default=float)

# Figures (blue/black only)
fig, ax = plt.subplots(figsize=(6, 4.2))
for k, ls in [("rf", "-"), ("iso", "--"), ("hybrid", "-.")]:
    for df, s, c, tag in [(test, s_te, NAVY, "known"), (novel, s_nv, BLUE, "novel")]:
        fpr, tpr, _ = roc_curve(df.label, s[k]); ax.plot(fpr, tpr, ls, color=c, lw=1.4, label=f"{k.upper()} ({tag})")
ax.plot([0, 1], [0, 1], ":", color="black", lw=.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC: known vs. novel attack variants")
ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig("fig_roc.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 3.8))
names = ["Rules", "Isolation Forest", "Random Forest", "Hybrid"]
x = np.arange(4)
ax.bar(x - .18, [res["test"][k]["f1"] for k in ["rules", "iso", "rf", "hybrid"]], .36, color=NAVY, label="Known attacks")
ax.bar(x + .18, [res["novel"][k]["f1"] for k in ["rules", "iso", "rf", "hybrid"]], .36, color=LIGHT, edgecolor=NAVY, label="Novel variant")
ax.set_xticks(x, names); ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
ax.set_title("Detection F1 by approach"); fig.tight_layout(); fig.savefig("fig_f1.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 3.8))
fi = list(res["feature_importance"].items())[:8][::-1]
ax.barh([f for f, _ in fi], [v for _, v in fi], color=BLUE); ax.set_xlabel("Gini importance")
ax.set_title("Top features (Random Forest)"); fig.tight_layout(); fig.savefig("fig_features.png", dpi=200)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.step(np.r_[0, manual], np.r_[0, np.arange(1, 7)], where="post", color="black", label="Manual SOC process")
ax.step(np.r_[0, auto], np.r_[0, np.arange(1, 7)], where="post", color=BLUE, lw=2, label="Automated playbook")
ax.set_xlabel("Minutes after detection"); ax.set_ylabel("Containment steps completed"); ax.set_xscale("symlog")
ax.legend(frameon=False); ax.set_title("Simulated containment timeline"); fig.tight_layout(); fig.savefig("fig_response.png", dpi=200)

print(json.dumps({k: res[k] for k in ["test", "novel", "stage_recall_hybrid", "counts", "confusion_hybrid"]}, indent=1))
print(res["response"]["auto_min"][-1], res["response"]["manual_min"][-1])
