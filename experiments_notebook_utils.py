import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)
import ef.models.classification as efc
import ef.dataset as ds

IDS = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
MAGS = ("VA", "VB", "VC", "CA", "CB", "CC", "CN", "EC", "AR", "AX", "AY", "AZ")
MAGSL = ("Voltage A", "Voltage B", "Voltage C", 
         "Current A", "Current B", "Current C", "Current N", 
         "Encoder Counts", "Accel. Ref.", "Accel. X", "Accel. Y", "Accel. Z")

FAILURES = {
    (0, 0, 0): "HLT",               # healthy
    (1, 0, 0): "UNB",               # unbalance
    (0, 0, 1): "BRB",               # broken bar
    (0, 1, 1): "MAL & BRB",         # misalignment and broken bar
    (0, 1, 0): "MAL",               # misalignment
    (1, 1, 0): "UNB & MAL",         # unbalance and misalignment
    (1, 1, 1): "UNB & MAL & BRB",   # unbalance, misalignment, and broken bar
    (1, 0, 1): "UNB & BRB"          # unbalance and broken bar
}

BIN_FAILURES = {
    (0, 0): "HLT",                 # healthy
    (0, 1): "BD"                   # bearing defect
}

ROOT_PATH = "/home/mariojg/research/datasets/motor_faults"

def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
        }

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_lines)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_lines)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_patches)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect(ax1, ax2, xmin, xmax, **kwargs):
    trans1 = blended_transform_factory(ax1.transData, ax1.transAxes)
    trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, trans1)
    mybbox2 = TransformedBbox(bbox, trans2)

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.1}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p

def text_labels(Y):
    return [FAILURES[tuple(y)] for y in Y]

def kaiser_test(eigenvalues):
    n_comps_to_retain = 0
    for ev in eigenvalues:
        if ev >= 1:
            n_comps_to_retain += 1
        else:
            break
    if n_comps_to_retain == 0:
        n_comps_to_retain = 0
    return n_comps_to_retain

def scree_test(eigenvalues):
    evs = eigenvalues.copy()
    evs *= float(len(evs))/max(evs)
    distances = [(x+y)/math.sqrt(2) for x, y in enumerate(evs)]
    return distances.index(min(distances))

def tests(pca):
    results = [["SCREE"], ["KAISER"]]
    for i in IDS:
        evs = pca[i].explained_variance_
        results[0].append(scree_test(evs))
        results[1].append(kaiser_test(evs))
    results_df = pd.DataFrame(data=results, columns=["test"]+list(MAGS))
    results_df = results_df.set_index("test")
    return results_df

def plot_eigenvalues_new(pca_ev, lim, fname):
    plt.figure(1, figsize=(4, 3), dpi=300)
    ax1 = plt.subplot(111)
    
    x = list(range(1,len(pca_ev)+1))
    scree = scree_test(pca_ev)
    kaiser = kaiser_test(pca_ev)
    
    ax1.scatter(x[:lim], pca_ev[:lim], s=3, color='k')
    ax1.axvline(scree, color='r', linestyle="--", linewidth=1, label="SCREE TEST")
    ax1.axvline(kaiser, color='orange', linewidth=1, label="KAISER RULE")
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    plt.savefig(f"../results/pca/figs/retain_study_{fname}.pdf", dpi=300, bbox_inches = "tight")
    plt.savefig(f"../results/pca/figs/retain_study_{fname}.tiff", dpi=300, bbox_inches = "tight")
    
    plt.show()

def plot_eigenvalues(pca):
    fig, axs = plt.subplots(4, 3, figsize=(7.48031, 9.5), dpi=300)
    axs = axs.flatten()
    for i in range(len(axs)):
        y = pca[IDS[i]].explained_variance_
        x = list(range(1,len(y)+1))
        scree = scree_test(y)
        kaiser = kaiser_test(y)
        lim = max((scree, kaiser))+200
        axs[i].scatter(x[:lim], y[:lim], s=1)
        axs[i].axvline(scree, color='r', linestyle="--", linewidth=0.5, label="SCREE TEST")
        axs[i].axvline(kaiser, color='orange', linewidth=0.5, label="KAISER RULE")
        axs[i].set_title(MAGSL[i], fontsize=10)
        axs[i].tick_params(axis='both', which='major', labelsize=7)
        axs[i].legend()
    plt.tight_layout()
    plt.show()
    
def models_evaluations(X, Y, tests_results):
    n_pc_comps_scree = list(tests_results.loc["SCREE"])
    n_pc_comps_kaiser = list(tests_results.loc["KAISER"])
    n_pc_comps_fixed = FIXED
    _, evals_scree = efc.build_pca_dt(X, Y, n_pc_comps_scree)
    _, evals_kaiser = efc.build_pca_dt(X, Y, n_pc_comps_kaiser)
    _, evals_fixed = efc.build_pca_dt(X, Y, n_pc_comps_fixed)
    return evals_scree, evals_kaiser, evals_fixed

def plot_multilabel_metrics(results):
    x = np.arange(10)
    titles = ("F1 Macro", "F1 Micro", "Accuracy", "Ranking Loss", "Hamming Loss", "Zero-One Loss")
    xlabs = ("0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.75", "1", "2", "5")
    fig, axs = plt.subplots(2, 3, figsize=(7.48031, 4), dpi=300, sharex=True)
    axs = axs.flatten()
    for i in range(len(axs)):
        mean = results[:,i,0]
        minv = results[:,i,1]
        maxv = results[:,i,2]
        axs[i].fill_between(x, minv, maxv, alpha=0.2)
        axs[i].plot(x, mean, marker="s", linewidth=1, markersize=3, mfc='none')
        axs[i].set_xticklabels(xlabs)
        axs[i].set_title(titles[i], fontsize=10)
        axs[i].tick_params(axis='both', which='major', labelsize=7)
    plt.xticks(x)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time interval (in seconds)", fontsize=10)
    plt.ylabel("Metric value", fontsize=10)

    plt.tight_layout()
    plt.show()
    
def plot_metric(results, metric_idx, fname, ylim=.5):
    x = np.arange(10)
    xlabs = ("0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.75", "1", "2", "5")
    
    mean = results[:,metric_idx,0]
    minv = results[:,metric_idx,1]
    maxv = results[:,metric_idx,2]
    
    fig = plt.figure(figsize=(4, 3), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.fill_between(x, minv, maxv, alpha=0.1, color='k', linewidth=0.0)
    ax.plot(x, mean, marker="s", linewidth=1, markersize=4, mfc='none', color='k')
    ax2.plot(x, mean, marker="s", linewidth=1, markersize=4, mfc='none', color='k')
    ax.axvline(4.5, linewidth=1, color='k', linestyle='--')
    ax2.axvline(4.5, linewidth=1, color='k', linestyle='--')
    ax2.set_xticklabels(xlabs)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(ylim, 1.)  # outliers only
    ax2.set_ylim(0, .2)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d*5, 1 + d*5), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d*5, 1 + d*5), **kwargs)  # bottom-right diagonal

    plt.xticks(x)
    
    plt.tight_layout()
    plt.savefig(f"../results/window_size/figs/{fname}.pdf", dpi=300)
    plt.savefig(f"../results/window_size/figs/{fname}.tiff", dpi=300)
    plt.show()
    
def plot_inverse_metric(results, metric_idx, fname, ylim=.3):
    x = np.arange(10)
    xlabs = ("0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.75", "1", "2", "5")
    
    mean = results[:,metric_idx,0]
    minv = results[:,metric_idx,1]
    maxv = results[:,metric_idx,2]
    
    fig = plt.figure(figsize=(4, 3), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax2.fill_between(x, minv, maxv, alpha=0.1, color='k', linewidth=0.0)
    ax2.plot(x, mean, marker="s", linewidth=1, markersize=4, mfc='none', color='k')
    ax.plot(x, mean, marker="s", linewidth=1, markersize=4, mfc='none', color='k')
    ax2.axvline(4.5, linewidth=1, color='k', linestyle='--')
    ax.axvline(4.5, linewidth=1, color='k', linestyle='--')
    ax2.set_xticklabels(xlabs)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(.8, 1)  # outliers only
    ax2.set_ylim(0, ylim)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d*5, +d*5), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d*5, +d*5), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.xticks(x)
    
    plt.tight_layout()
    plt.savefig(f"../results/window_size/figs/{fname}.pdf", dpi=300)
    plt.savefig(f"../results/window_size/figs/{fname}.tiff", dpi=300)
    plt.show()
    
def predictions_confusion_matrices(X, Y, classifier, n_kfolds, random_state=1):
    def split_training_test(X_in, train_idxs, test_idxs):
        X_train, X_test = (None, None)
        if type(X_in) is list:
            X_train, X_test = ([], [])
            for X_i in X_in:
                X_train.append(X_i[train_idxs])
                X_test.append(X_i[test_idxs])
        else:
            X_train = X_in[train_idxs]
            X_test = X_in[test_idxs]
        return tuple((X_train, X_test))
    
    classes=[FAILURES[k] for k in FAILURES]
    cm = np.zeros((8,8))
    kf = KFold(n_kfolds, True, random_state)
    folds = kf.split(X[0] if type(X) is list else X)
    for train_idxs, test_idxs in folds:
        fold_classifier = clone(classifier)
        x_train, x_test = split_training_test(X, train_idxs, test_idxs)
        y_train, y_test = Y[train_idxs], Y[test_idxs]

        fold_classifier.fit(x_train, y_train)
        y_pred = fold_classifier.predict(x_test)
        fold_cnf_matrix = confusion_matrix(text_labels(y_test), text_labels(y_pred), labels=classes)
        cm = np.add(cm, fold_cnf_matrix)
    return np.round(cm).astype('int')

def plot_confusion_matrix(cm, failures, fname):
    classes=[failures[k] for k in failures]

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    thresh = cm.max() / 2.
    for j, k in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(k, j, format(cm[j, k], "d"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[j, k] > thresh else "black")

    fig.colorbar(im)
    plt.savefig(f"../results/confusion_matrices/figs/{fname}.pdf", dpi=300, bbox_inches = "tight")
    plt.savefig(f"../results/confusion_matrices/figs/{fname}.tiff", dpi=300, bbox_inches = "tight")
    plt.show()