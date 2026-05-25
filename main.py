import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

df = pd.read_csv('1stproject.csv')
df = df[df['insurance_status'] != 'Other']  # exclude "uninsured" patients
df = df.drop_duplicates()
df = df.sample(frac=0.60, random_state=9197) # use 60% of database
df = df.dropna(subset=['age']) # mandatory: age defines every subset split (dep_name has no missing values)
df = df.reset_index(drop=True) # reset row indices
df['disposition'] = (df['disposition'] == 'Admit').astype(int)  # target: 1=Admit, 0=Discharge

y = df['disposition']
X = df.drop(columns=['disposition']) # exclude target
# store original columns for subgroup creation
age_col = df['age'] 
dep_col = df['dep_name']
esi_col = df['esi'] # use for ESI-weighted AUC
X = pd.get_dummies(X, drop_first=True) # one-hot encoding + avoid multicollinearity
X = X.dropna(axis=1, how='all') # remove columns full of NAs

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9197) # non-temporal 80/20 split

# align age/dep values with train/test split
age_train = age_col.loc[X_train.index]
dep_train = dep_col.loc[X_train.index]
age_test  = age_col.loc[X_test.index]
dep_test  = dep_col.loc[X_test.index]

# assign higher evaluation weight to high-acuity ESI cases
esi_weights_test = esi_col.loc[X_test.index].map({1: 5, 2: 4, 3: 2, 4: 1, 5: 1}).fillna(1) 

# identify high-missing columns (>50%)
high_missing_cols = [col for col in X_train.columns 
                     if X_train[col].isna().sum() / len(X_train) > 0.50]

# vital signs: median imputation 
vital_signs = ['spo2_last', 'temp_last']

# create binary "was_tested" flag instead of imputing
lab_missing_cols = [col for col in high_missing_cols if col not in vital_signs]

# build "was_tested" flags
tested_train = pd.DataFrame(
    {f'{col}_tested': X_train[col].notna().astype(int) for col in lab_missing_cols},
    index=X_train.index
)
tested_test = pd.DataFrame(
    {f'{col}_tested': X_test[col].notna().astype(int) for col in lab_missing_cols},
    index=X_test.index
)
X_train = pd.concat([X_train.drop(columns=lab_missing_cols), tested_train], axis=1)
X_test  = pd.concat([X_test.drop(columns=lab_missing_cols),  tested_test],  axis=1)


# replace NA values with the median
imputer = SimpleImputer(strategy='median').set_output(transform="pandas") 
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# remove features with very low variance
selector = VarianceThreshold(threshold=0.01)
X_train_var = selector.fit_transform(X_train)
X_test_var  = selector.transform(X_test)

selected_cols = X_train.columns[selector.get_support()] # keep the names of selected features
X_train_var = pd.DataFrame(X_train_var, columns=selected_cols, index=X_train.index)
X_test_var  = pd.DataFrame(X_test_var,  columns=selected_cols, index=X_test.index)

corr_matrix = X_train_var.corr().abs()  # compute absolute Pearson correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # keep upper triangle only
to_drop = [col for col in upper.columns if any(upper[col] > 0.90)] # identify highly correlated features
# remove these features
X_train_var = X_train_var.drop(columns=to_drop)
X_test_var  = X_test_var.drop(columns=to_drop)

# feature importance ranking with RF
rf_fs = RandomForestClassifier(n_estimators=100, max_depth=8,
                                min_samples_leaf=20, random_state=9197, n_jobs=-1)
rf_fs.fit(X_train_var, y_train)

feat_imp = pd.DataFrame({'feature': X_train_var.columns,
                         'importance': rf_fs.feature_importances_})\
                         .sort_values('importance', ascending=False) # rank by importance score
feat_imp['cumulative'] = feat_imp['importance'].cumsum()

# visualize top 15 RF-selected features
fig_fi, ax_fi = plt.subplots(figsize=(8, 6))

ax_fi.barh(
    feat_imp.head(15).sort_values('importance')['feature'],
    feat_imp.head(15).sort_values('importance')['importance'],
    color="#185FA5",
    alpha=0.85
)

ax_fi.set_xlabel("Importance Score")
ax_fi.set_ylabel("Feature")
ax_fi.set_title("Top 15 Features - RF Feature Selection", fontweight="bold")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=180,
            bbox_inches="tight", facecolor="white")
plt.show()
plt.close(fig_fi)

# keep features explaining 90% of total importance
n_90 = (feat_imp['cumulative'] <= 0.90).sum() + 1 
top_features = feat_imp.head(n_90)['feature'].tolist()

X_train_final = X_train_var[top_features]   
X_test_final  = X_test_var[top_features]

print(f"Final features: {X_train_final.shape[1]}") # number of final features
print(f"Train: {X_train_final.shape[0]} | Test: {X_test_final.shape[0]}") # final sample sizes

def get_mask(age_series, dep_series, age_range, dep):
    age_mask = (age_series >= age_range[0]) & (age_series <= age_range[1]) # filter patients by age range
    return age_mask & (dep_series == dep) # keep only patients from the selected department

# define specialist masks for each demographic subset
m1 = get_mask(age_train, dep_train, (18, 44),  'A')
m2 = get_mask(age_train, dep_train, (18, 44),  'B')
m3 = get_mask(age_train, dep_train, (45, 64),  'A')
m4 = get_mask(age_train, dep_train, (45, 64),  'B')
m5 = get_mask(age_train, dep_train, (65, 120), 'A')
m6 = get_mask(age_train, dep_train, (65, 120), 'B')

def make_abd_mask(raw_df, index):
    sub = raw_df.loc[index]

    # define the chief complaint (cc) features
    cols_needed = ['cc_abdominalpain', 'cc_flankpain', 'cc_backpain', 'cc_emesis']

    # initialize a boolean mask of False values matching the subset's index length
    mask = pd.Series(False, index=sub.index)
    for col in cols_needed:
        if col in sub.columns:
            mask = mask | (sub[col].fillna(0) == 1) # True if any of the 4 cc is present
    return mask

abd_mask_train = make_abd_mask(df, X_train.index)

# store the 6 demographic + 1 complaint-based masks in a list
masks = [m1, m2, m3, m4, m5, m6, abd_mask_train]

# each name encodes the algorithm and its training subset
names = ['HGB(18-44,A)', 'RF(18-44,B)',
         'DT(45-64,A)',  'NB(45-64,B)',
         'HGB_cons(65+,A)', 'RF(65+,B)',
         'CatBoost(Abd)']


# define one specialist base classifier per subset
clf1 = HistGradientBoostingClassifier(max_iter=106, max_depth=6, learning_rate=0.101, random_state=9197)
clf2 = RandomForestClassifier(n_estimators=67, max_depth=6, random_state=9197, n_jobs=-1)
clf3 = DecisionTreeClassifier(max_depth=5, min_samples_leaf=49, random_state=9197)
clf4 = GaussianNB(var_smoothing=3.07e-08)
clf5 = HistGradientBoostingClassifier(max_iter=106, max_depth=3, learning_rate=0.046, random_state=9197) # conservative
clf6 = RandomForestClassifier(n_estimators=142, max_depth=7, random_state=9197, n_jobs=-1)
clf7 = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_seed=9197, verbose=0)

clfs = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]

# stratified 5-fold CV to preserve class balance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9197)

# initialise out-of-fold prediction matrix: [n_train_samples x n_classifiers]
oof_preds = np.zeros((len(X_train_final), len(clfs)))

# train each specialist classifier on its subset via 5-fold CV and collect out-of-fold predictions for the meta-classifier
for j, (clf, mask) in enumerate(zip(clfs, masks)):
    for train_idx, test_idx in cv.split(X_train_final, y_train):
        train_subset_idx = train_idx[mask.iloc[train_idx].values]
        clf_clone = clone(clf)
        clf_clone.fit(X_train_final.iloc[train_subset_idx],
                      y_train.iloc[train_subset_idx])
        oof_preds[test_idx, j] = clf_clone.predict_proba(
            X_train_final.iloc[test_idx])[:, 1]

# build meta-classifier training set from out-of-fold predictions
meta_X_train = pd.DataFrame(oof_preds, columns=names,
                              index=X_train_final.index)

# refit each base classifier on its full subset 
for clf, mask in zip(clfs, masks):
    clf.fit(X_train_final[mask], y_train[mask])

# train meta-classifier on out-of-fold predictions
meta_clf = LogisticRegression(C=0.026, random_state=9197)
meta_clf.fit(meta_X_train, y_train)

meta_X_test = pd.DataFrame(
    np.column_stack([clf.predict_proba(X_test_final)[:, 1] for clf in clfs]),
    columns=names, index=X_test_final.index
)

# generate final predicted probabilities
y_pred_prob = meta_clf.predict_proba(meta_X_test)[:, 1]
y_pred = (y_pred_prob >= 0.30).astype(int) # threshold to classify as Admit/Discharge

# stacking performance on test set
print("\n" + "="*50)
print("Stacking Results")
print("="*50)
print(f"AUC:  {roc_auc_score(y_test, y_pred_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Discharge', 'Admit']))

# re-evaluate giving higher weight to high-acuity (ESI 1-2) patients
print(f"Weighted AUC (ESI): {roc_auc_score(y_test, y_pred_prob, sample_weight=esi_weights_test):.4f}")
print(classification_report(y_test, y_pred, sample_weight=esi_weights_test, target_names=['Discharge', 'Admit']))

# compare train vs test AUC to detect overfitting
train_pred = meta_clf.predict_proba(meta_X_train)[:, 1]
test_pred  = meta_clf.predict_proba(meta_X_test)[:, 1]
print(f"\nOverfit Check:")
print(f"  Train AUC: {roc_auc_score(y_train, train_pred):.4f}")
print(f"  Test AUC:  {roc_auc_score(y_test,  test_pred):.4f}")
print(f" Difference:   {roc_auc_score(y_train, train_pred) - roc_auc_score(y_test, test_pred):.4f}")


print("\nIndividual AUC per base classifier:")
for clf, mask, name in zip(clfs, masks, names):
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test_final)[:, 1])
    print(f"  {name:20s} | subset: {mask.sum():5d} | AUC: {auc_score:.4f}")


# =============================================================================
# Explainability of the Meta-Learner (SHAP)
# =============================================================================



explainer = shap.LinearExplainer(meta_clf, meta_X_train)
shap_vals = explainer(meta_X_test)

plt.figure(figsize=(8, 4))
shap.summary_plot(shap_vals, meta_X_test,
                  feature_names=meta_X_test.columns,
                  show=False)
plt.title("SHAP — Base Classifier Contribution to Meta-Decision",
          fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()
plt.close()
print("Saved: shap_summary.png")


# =============================================================================
# VALIDATION SET
# =============================================================================

# load external validation set (Dep C) and apply the same preprocessing pipeline
df_val = pd.read_csv('1stproject-TestSet.csv')
df_val = df_val.dropna(subset=['age'])
df_val = df_val.reset_index(drop=True)
df_val = df_val[df_val['insurance_status'] != 'Other'] 
df_val['disposition'] = (df_val['disposition'] == 'Admit').astype(int)

y_val = df_val['disposition']
X_val = df_val.drop(columns=['disposition'])
X_val = pd.get_dummies(X_val, drop_first=True)
X_val = X_val.dropna(axis=1, how='all')


X_val = X_val.reindex(columns=X.columns, fill_value=0)

tested_val = pd.DataFrame(
    {f'{col}_tested': X_val[col].notna().astype(int) for col in lab_missing_cols},
    index=X_val.index
)
X_val = pd.concat([X_val.drop(columns=lab_missing_cols, errors='ignore'), tested_val], axis=1)

# apply fitted transformers (no refit)
X_val = imputer.transform(X_val)
X_val = pd.DataFrame(X_val, columns=X_train.columns)
X_val_var = selector.transform(X_val)
X_val_var = pd.DataFrame(X_val_var, columns=selected_cols)
X_val_var = X_val_var.drop(columns=to_drop, errors='ignore')
X_val_final = X_val_var[top_features]

meta_X_val = pd.DataFrame(
    np.column_stack([clf.predict_proba(X_val_final)[:, 1] for clf in clfs]),
    columns=names
)

# apply lower threshold (0.20) 
y_val_prob = meta_clf.predict_proba(meta_X_val)[:, 1]
y_val_pred = (y_val_prob >= 0.20).astype(int) 

print("\n" + "="*50)
print("Validation Set Results")
print("="*50)
print(f"AUC:  {roc_auc_score(y_val, y_val_prob):.4f}")
print(classification_report(y_val, y_val_pred, target_names=['Discharge', 'Admit']))

# re-evaluate giving higher weight to high-acuity (ESI 1-2) patients
esi_weights_val = df_val['esi'].map({1: 5, 2: 4, 3: 2, 4: 1, 5: 1}).fillna(1)
print(f"Weighted AUC (ESI): {roc_auc_score(y_val, y_val_prob, sample_weight=esi_weights_val):.4f}")
print(classification_report(y_val, y_val_pred, sample_weight=esi_weights_val, target_names=['Discharge', 'Admit']))


# =============================================================================
# MODEL EVALUATION DASHBOARD
# =============================================================================


# colour palette 
C_BLUE   = "#185FA5"
C_GREEN  = "#1D9E75"
C_ORANGE = "#D85A30"
C_PURPLE = "#534AB7"
C_GRAY   = "#888780"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# figure  (3x3 grid)

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.45, wspace=0.35,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.07,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

fig.suptitle(
    "Stacking Ensemble — Full Model Evaluation",
    fontsize=16, fontweight="bold", color="#1a1a1a", y=0.97
)


# helper 
def plot_cm(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Discharge", "Admit"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    total = cm.sum()
    for text_obj, val in zip(ax.texts, cm.flatten()):
        text_obj.set_text(f"{val:,}\n({val/total*100:.1f}%)")
        text_obj.set_fontsize(8)


# 1. ROC TEST
fpr_t, tpr_t, _ = roc_curve(y_test, y_pred_prob)
auc_t = auc(fpr_t, tpr_t)
ax1.plot(fpr_t, tpr_t, color=C_BLUE, lw=2, label=f"AUC = {auc_t:.4f}")
ax1.plot([0,1],[0,1], "--", color=C_GRAY, lw=1)
ax1.fill_between(fpr_t, tpr_t, alpha=0.08, color=C_BLUE)
ax1.set(xlim=[0,1], ylim=[0,1.02],
        xlabel="False Positive Rate", ylabel="True Positive Rate")
ax1.set_title("① ROC Curve — Test Set (A+B)", fontsize=10, fontweight="bold")
ax1.legend(fontsize=9, loc="lower right")


# 2. ROC VALIDATION 
fpr_v, tpr_v, _ = roc_curve(y_val, y_val_prob)
auc_v = auc(fpr_v, tpr_v)
ax2.plot(fpr_v, tpr_v, color=C_ORANGE, lw=2, label=f"AUC = {auc_v:.4f}")
ax2.plot([0,1],[0,1], "--", color=C_GRAY, lw=0.5)
ax2.fill_between(fpr_v, tpr_v, alpha=0.08, color=C_ORANGE)
ax2.set(xlim=[0,1], ylim=[0,1.02],
        xlabel="False Positive Rate", ylabel="True Positive Rate")
ax2.set_title("② ROC Curve — Validation Set (C)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=9, loc="lower right")


# 3. AUC PER BASE CLASSIFIER (ascending)
base_aucs     = [roc_auc_score(y_test, clf.predict_proba(X_test_final)[:, 1])
                 for clf in clfs]
meta_auc_test = roc_auc_score(y_test, y_pred_prob)

base_labels = [n.replace("(", "\n(") for n in names]
colors_bar  = [C_BLUE, C_GREEN, C_BLUE, C_GREEN, C_BLUE, C_GREEN, C_PURPLE]

# sort ascending so the best classifier appears at the top
sorted_pairs  = sorted(zip(base_aucs, base_labels, colors_bar), key=lambda x: x[0])
s_aucs, s_labels, s_colors = zip(*sorted_pairs)

bars = ax3.barh(s_labels, s_aucs, color=s_colors, alpha=0.85, height=0.55)
ax3.axvline(meta_auc_test, color=C_PURPLE, lw=2, linestyle="--",
            label=f"Meta Stacking = {meta_auc_test:.4f}")
for bar, v in zip(bars, s_aucs):
    ax3.text(v + 0.0003, bar.get_y() + bar.get_height()/2,
             f"{v:.4f}", va="center", fontsize=8)

xmin = min(s_aucs) - 0.005
ax3.set_xlim([xmin, meta_auc_test + 0.012])
ax3.set_xlabel("AUC", fontsize=9)
ax3.set_title("③ AUC per Base Classifier", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0)


# 4. LEARNING CURVE 
train_sizes, train_scores, val_scores = learning_curve(
    clone(meta_clf), meta_X_train, y_train,
    cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1
)
tr_mean  = train_scores.mean(axis=1)
tr_std   = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std  = val_scores.std(axis=1)

ax4.plot(train_sizes, tr_mean,  "o-", color=C_BLUE,   lw=2, label="Train AUC")
ax4.plot(train_sizes, val_mean, "s-", color=C_ORANGE,  lw=2, label="CV AUC")
ax4.fill_between(train_sizes, tr_mean-tr_std,  tr_mean+tr_std,
                 alpha=0.12, color=C_BLUE)
ax4.fill_between(train_sizes, val_mean-val_std, val_mean+val_std,
                 alpha=0.12, color=C_ORANGE)
ax4.set_xlabel("Training Set Size", fontsize=9)
ax4.set_ylabel("AUC", fontsize=9)
ax4.set_title("④ Learning Curve (Meta-Classifier)", fontsize=10, fontweight="bold")
ax4.legend(fontsize=9)
ax4.tick_params(axis='x', labelsize=7)

## 5. BIAS-VARIANCE TRADEOFF

C_values = [0.001,0.005,0.01,0.026,0.05,0.1,0.5,1,10,100]

train_aucs_bv = []
cv_aucs_bv = []

for c in C_values:

    clf_temp = LogisticRegression(C=c, random_state=9197)

    clf_temp.fit(meta_X_train, y_train)

    train_auc_temp = roc_auc_score(
        y_train,
        clf_temp.predict_proba(meta_X_train)[:,1]
    )

    cv_auc_temp = cross_val_score(
        clf_temp,
        meta_X_train,
        y_train,
        cv=5,
        scoring='roc_auc'
    ).mean()

    train_aucs_bv.append(train_auc_temp)
    cv_aucs_bv.append(cv_auc_temp)

x_idx = range(len(C_values))

ax5.plot(x_idx, train_aucs_bv,
         "o-", color=C_BLUE, lw=2,
         label="Train AUC")

ax5.plot(x_idx, cv_aucs_bv,
         "s-", color=C_ORANGE, lw=2,
         label="CV AUC")

ax5.axvline(
    C_values.index(0.026),
    color=C_GREEN,
    linestyle="--",
    lw=1.2,
    alpha=0.8,
    label="Selected C=0.026"
)

ax5.set_xticks(list(x_idx))

ax5.set_xticklabels(
    [str(c) for c in C_values],
    rotation=45,
    fontsize=7
)

ax5.set_xlabel("Model Complexity (Regularization C)", fontsize=9)
ax5.set_ylabel("AUC", fontsize=9)

ax5.set_title(
    "⑤ Bias-Variance Tradeoff",
    fontsize=10,
    fontweight="bold"
)

ax5.legend(fontsize=8)

ax5.set_ylim([
    min(cv_aucs_bv)-0.0005,
    max(train_aucs_bv)+0.0005
])



# 6. OVERFIT BAR
train_auc = roc_auc_score(y_train, meta_clf.predict_proba(meta_X_train)[:, 1])
test_auc  = roc_auc_score(y_test,  meta_clf.predict_proba(meta_X_test)[:, 1])
val_auc   = roc_auc_score(y_val,   meta_clf.predict_proba(meta_X_val)[:, 1])

sets   = ["Train\n(A+B)", "Test\n(A+B)", "Validation\n(C)"]
aucs_v = [train_auc, test_auc, val_auc]
cols_v = [C_BLUE, C_GREEN, C_ORANGE]

b = ax6.bar(sets, aucs_v, color=cols_v, alpha=0.85, width=0.5)
for bar, v in zip(b, aucs_v):
    ax6.text(bar.get_x() + bar.get_width()/2, v + 0.001,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax6.set_ylim([0.85, 0.93])
ax6.set_ylabel("AUC", fontsize=9)
ax6.set_title("⑥ Train / Test / Validation AUC\n(Overfitting Check)",
              fontsize=10, fontweight="bold")

gap = train_auc - test_auc
ax6.annotate(f"Gap = {gap:+.4f}",
             xy=(1, test_auc), xytext=(1.45, test_auc + 0.01),
             arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1),
             fontsize=8, color=C_GRAY)


# 7 & 8. CONFUSION MATRICES 
plot_cm(ax7, y_test, y_pred,     "⑦ Confusion Matrix — Test (A+B)")
plot_cm(ax8, y_val,  y_val_pred, "⑧ Confusion Matrix — Validation (C)")


# 9. PRECISION-RECALL CURVE 
prec_t, rec_t, _ = precision_recall_curve(y_test, y_pred_prob)
ap_t = average_precision_score(y_test, y_pred_prob)
ax9.plot(rec_t, prec_t, color=C_BLUE,   lw=2, label=f"Test (A+B)  AP={ap_t:.3f}")

prec_v, rec_v, _ = precision_recall_curve(y_val, y_val_prob)
ap_v = average_precision_score(y_val, y_val_prob)
ax9.plot(rec_v, prec_v, color=C_ORANGE, lw=2, label=f"Validation (C)  AP={ap_v:.3f}")

baseline_t = y_test.mean()
baseline_v = y_val.mean()
ax9.axhline(baseline_t, color=C_BLUE,   lw=1, linestyle=":", alpha=0.6,
            label=f"Baseline Test = {baseline_t:.2f}")
ax9.axhline(baseline_v, color=C_ORANGE, lw=1, linestyle=":", alpha=0.6,
            label=f"Baseline Val  = {baseline_v:.2f}")

ax9.set_xlabel("Recall", fontsize=9)
ax9.set_ylabel("Precision", fontsize=9)
ax9.set_title("⑨ Precision-Recall Curve\n(Critical with class imbalance)",
              fontsize=10, fontweight="bold")
ax9.set_xlim([0, 1])
ax9.set_ylim([0, 1.02])
ax9.legend(fontsize=8, loc="upper right")


# save 
plt.savefig("model_dashboard.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved: model_dashboard.png")




