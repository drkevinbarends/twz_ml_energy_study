import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from fourClassModelArchitecture import Model

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hyperparameter_file", required=True)

    return parser.parse_args()


# ---------------------------------------------------------
# Setup Data
# ---------------------------------------------------------

def load_data(input_file):

    df = pd.read_csv(input_file)
    df["eventWeights"] = abs(df["eventWeights"])

    features = [
        # Z
        "ZCandidate0_pt_NOSYS",
        "ZCandidate0_eta_NOSYS",
        "ZCandidate0_mass_NOSYS",

        # non-Z leptons
        "nonZ_lep0_pt_NOSYS","nonZ_lep0_eta_NOSYS",
        "nonZ_lep1_pt_NOSYS","nonZ_lep1_eta_NOSYS",

        # bjets
        "bjet0_pt_NOSYS","bjet0_eta_NOSYS",
        "bjet1_pt_NOSYS","bjet1_eta_NOSYS",

        # non-bjet
        "non_bjet0_pt_NOSYS","non_bjet0_eta_NOSYS",

        # global
        "nJets_NOSYS","nBjets_NOSYS",
        "HT_NOSYS","SMT_NOSYS",
        "met_met_GeV_NOSYS",
        "sumBjetPt_NOSYS","sumZCandidatePt_NOSYS",

        # tagging
        "non_bjet0_btag_score_NOSYS",

        # physics
        "dR_Z_l_min_NOSYS","dR_Z_l_max_NOSYS",
        "m_bl_min_NOSYS","m_bl_second_min_NOSYS",
        "dPhi_met_Z_NOSYS","dPhi_met_nonZ_lep0_NOSYS","dPhi_met_nonZ_lep1_NOSYS",
        "dPhi_met_bjet0_phi_NOSYS","dPhi_met_bjet1_phi_NOSYS",
        "mT_lep_met_min_NOSYS","mT_lep_met_max_NOSYS",
        "dR_nonZ_leps_NOSYS",
        "dR_Zlep_nonZlep_min_NOSYS",
        "m_bb_NOSYS","dR_bb_NOSYS",
        "dR_bl_min_NOSYS","dR_bl_secondmin_NOSYS"
    ]

    X = df[features].values
    y = df["label"].values
    w = df["eventWeights"].values
    splits = df["eventNumber"].values % 4

    return X, y, w, splits


# ---------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------

def plot_classifier_all_processes(train_prob, test_prob, y_train, y_test, w_train, w_test, class_idx, output_file):

    names = ["other","ZZ","ttZ","tWZ"]
    bins = np.linspace(0,1,50)

    plt.figure(figsize=(8,6))

    for i, pname in enumerate(names):

        mask_train = y_train == i
        mask_test  = y_test == i

        plt.hist(train_prob[mask_train,class_idx], bins=bins,
                 weights=w_train[mask_train], density=True,
                 histtype="step", linestyle="-",
                 label=f"{pname} (train)")

        plt.hist(test_prob[mask_test,class_idx], bins=bins,
                 weights=w_test[mask_test], density=True,
                 histtype="step", linestyle="--",
                 label=f"{pname} (test)")

    plt.xlabel(f"Classifier score ({names[class_idx]})")
    plt.ylabel("Normalized events")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_classifier_stacked_test(test_prob, y_test, w_test, class_idx, output_file):

    names = ["other","ZZ","ttZ","tWZ"]
    bins = np.linspace(0,1,50)

    score_list = []
    weight_list = []

    for i in range(4):
        mask = y_test == i
        score_list.append(test_prob[mask, class_idx])
        weight_list.append(w_test[mask])

    plt.figure(figsize=(8,6))

    plt.hist(score_list, bins=bins, weights=weight_list,
             stacked=True, label=names)

    plt.xlabel(f"Classifier score ({names[class_idx]})")
    plt.ylabel("Events (weighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_score_sets(score, y, weights, output_prefix):

    process_names = ["other","ZZ","ttZ","tWZ"]
    bins = np.linspace(0,1,50)

    # NORMALIZED
    plt.figure(figsize=(8,6))

    for i, pname in enumerate(process_names):
        mask = (y == i) & (score >= 0)

        if np.sum(weights[mask]) == 0:
            continue

        w_norm = weights[mask] / np.sum(weights[mask])

        plt.hist(score[mask], bins=bins, weights=w_norm,
                 histtype="step", label=pname)

    plt.xlabel("Score")
    plt.ylabel("Normalized events")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix + "_normalized.png")
    plt.close()

    # STACKED
    plt.figure(figsize=(8,6))

    score_list = []
    weight_list = []

    for i in range(4):
        mask = (y == i) & (score >= 0)
        score_list.append(score[mask])
        weight_list.append(weights[mask])

    plt.hist(score_list, bins=bins, weights=weight_list,
             stacked=True, label=process_names)

    plt.xlabel("Score")
    plt.ylabel("Events (weighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_prefix + "_stacked.png")
    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.hyperparameter_file) as f:
        params = json.load(f)

    X, y, w_events, splits = load_data(args.input_file)

    for fold in range(4):

        tf.keras.backend.clear_session()
        logging.info(f"Running fold {fold}")

        test_mask = splits == fold
        val_mask  = splits == ((fold+3) % 4)
        train_mask = (~test_mask) & (~val_mask)

        X_train = X[train_mask]
        y_train = y[train_mask]
        w_train = w_events[train_mask]
        w_train[y_train == 3] *= 5

        X_val = X[val_mask]
        y_val = y[val_mask]
        w_val = w_events[val_mask]
        w_val[y_val == 3] *= 5

        X_test = X[test_mask]
        y_test = y[test_mask]
        w_test = w_events[test_mask]

        # -------------------------------------------------
        # Directory structure (RESTORED)
        # -------------------------------------------------

        model_dir = os.path.join(args.output_dir, "trained")
        os.makedirs(model_dir, exist_ok=True)

        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        model = Model(
            in_shape=X_train.shape[1],
            x_train=X_train,
            learning_rate=params["learning_rate"],
            layer_sizes=params["layer_sizes"],
            dropout_rate=params["dropout_rate"],
            use_batchnorm=params["use_batchnorm"],
        )

        model.train(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            class_weight_dict,
            batch_size=params["batch_size"],
            epochs=100,
            output_dir=fold_dir
        )

        model.save_model(fold_dir)

        train_prob = model.model.predict(X_train)
        test_prob  = model.model.predict(X_test)

        # -------------------------------------------------
        # Classifier plots
        # -------------------------------------------------

        for c in range(4):

            plot_classifier_all_processes(
                train_prob, test_prob,
                y_train, y_test,
                w_train, w_test,
                c,
                os.path.join(fold_dir, f"classifier_all_{c}.png")
            )

            plot_classifier_stacked_test(
                test_prob, y_test, w_test,
                c,
                os.path.join(fold_dir, f"classifier_stacked_test_{c}.png")
            )

        # -------------------------------------------------
        # Scores
        # -------------------------------------------------

        p_other = test_prob[:,0]
        p_ZZ    = test_prob[:,1]
        p_ttZ   = test_prob[:,2]
        p_tWZ   = test_prob[:,3]

        score_ratio = p_tWZ / (p_tWZ + p_ttZ + 1e-6)

        is_tWZ_cat = (p_tWZ > p_ttZ) & (p_tWZ > p_ZZ) & (p_tWZ > p_other)
        score_tWZ_cat = np.where(is_tWZ_cat, p_tWZ, -1.0)

        is_ttZ_cat = (p_ttZ > p_tWZ) & (p_ttZ > p_ZZ) & (p_ttZ > p_other)
        score_ttZ_cat = np.where(is_ttZ_cat, p_ttZ, -1.0)

        plot_score_sets(score_ratio, y_test, w_test, os.path.join(fold_dir,"ratio_score"))
        plot_score_sets(score_tWZ_cat, y_test, w_test, os.path.join(fold_dir,"tWZ_category"))
        plot_score_sets(score_ttZ_cat, y_test, w_test, os.path.join(fold_dir,"ttZ_category"))


if __name__ == "__main__":
    main()