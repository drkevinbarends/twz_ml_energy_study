import argparse
import logging
import os
import pandas as pd
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ROOT files for tWZ ML analysis.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--replot", action='store_true', help="Flag to re-plot histograms.")
    return parser.parse_args()

def plot_histogram(df, column_name, weight_column, output_dir):
    """Plots and saves histograms."""
    os.makedirs(f"{output_dir}/kinematic_distributions", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    tWZ_condition = (df['label'] == 3)
    ttZ_condition = (df['label'] == 2)
    ZZ_condition = (df['label'] == 1)
    other_condition = (df['label'] == 0)

    tWZ_data = df[tWZ_condition][column_name]
    ttZ_data = df[ttZ_condition][column_name]
    ZZ_data = df[ZZ_condition][column_name]
    other_data = df[other_condition][column_name]

    # Define common bins based on combined min and max
    bins = np.linspace(
        min(tWZ_data.min(), ttZ_data.min(), ZZ_data.min(), other_data.min()),
        max(tWZ_data.max(), ttZ_data.max(), ZZ_data.max(), other_data.max()),
        51  # 50 bins means 51 edges
    )

    plt.hist(tWZ_data, bins=bins, alpha=0.5, label='tWZ', color='blue',
         weights=df[tWZ_condition][weight_column])
    plt.hist(ttZ_data, bins=bins, alpha=0.5, label='ttZ', color='green',
         weights=df[ttZ_condition][weight_column])
    plt.hist(ZZ_data, bins=bins, alpha=0.5, label='ZZ', color='yellow',
         weights=df[ZZ_condition][weight_column])
    plt.hist(other_data, bins=bins, alpha=0.5, label='Other', color='red',
         weights=df[other_condition][weight_column])
    
    plt.xlabel(column_name)
    plt.ylabel('Weighted Frequency')
    plt.title(f'{column_name} Distribution (Weighted)')
    plt.legend()
    plt.savefig(f'{output_dir}/kinematicDistributions/{column_name}.png')
    plt.close()

def process_files(input_dir, output_dir, replot):
    """Main data processing function."""
    tWZ_files = ["tWZ.root"]
    ttZ_files = ["ttZ.root"]
    ZZ_files = ["ZZ.root"]
    other_files = ["other.root"]
    
    features = [
        "eventNumber",
        "mcChannelNumber",
        "xSection",
        "lumi",
        "weight_total_NOSYS",
        "sumWeights_NOSYS",
        "nJets_NOSYS",
        "nBjets_NOSYS",
        "SMT_NOSYS",
        "HT_NOSYS",
        "LT_NOSYS",
        "ST_NOSYS",
        "met_met_GeV_NOSYS",
        "met_phi_NOSYS",
        "ZCandidate0_pt_NOSYS",
        "ZCandidate0_eta_NOSYS",
        "ZCandidate0_phi_NOSYS",
        "ZCandidate0_mass_NOSYS",
        "lep0_pt_NOSYS",
        "lep0_eta_NOSYS",
        "lep0_phi_NOSYS",
        "lep1_pt_NOSYS",
        "lep1_eta_NOSYS",
        "lep1_phi_NOSYS",
        "lep2_pt_NOSYS",
        "lep2_eta_NOSYS",
        "lep2_phi_NOSYS",
        "lep3_pt_NOSYS",
        "lep3_eta_NOSYS",
        "lep3_phi_NOSYS",
        "jet0_pt_NOSYS",
        "jet0_eta_NOSYS",
        "jet0_phi_NOSYS",
        "jet1_pt_NOSYS",
        "jet1_eta_NOSYS",
        "jet1_phi_NOSYS",
        "nonZ_lep0_pt_NOSYS",
        "nonZ_lep0_eta_NOSYS",
        "nonZ_lep0_phi_NOSYS",
        "nonZ_lep1_pt_NOSYS",
        "nonZ_lep1_eta_NOSYS",
        "nonZ_lep1_phi_NOSYS",
        "bjet0_pt_NOSYS",
        "bjet0_eta_NOSYS",
        "bjet0_phi_NOSYS",
        "bjet1_pt_NOSYS",
        "bjet1_eta_NOSYS",
        "bjet1_phi_NOSYS",
        "non_bjet0_pt_NOSYS",
        "non_bjet0_eta_NOSYS",
        "non_bjet0_phi_NOSYS",
        "non_bjet1_pt_NOSYS",
        "non_bjet1_eta_NOSYS",
        "non_bjet1_phi_NOSYS",
        "sumBjetPt_NOSYS",
        "sumZCandidatePt_NOSYS",
        "non_bjet0_btag_score_NOSYS",
        "LT_MET_pt_NOSYS",
        "allLep_pt_NOSYS",
        "allLep_eta_NOSYS",
        "allLep_phi_NOSYS",
        "allLep_mass_NOSYS",
        "deltaPhi_Lep0_MET_NOSYS",
        "deltaPhi_Lep1_MET_NOSYS",
        "deltaPhi_Lep2_MET_NOSYS",
        "deltaPhi_Lep3_MET_NOSYS",
        "allLepBjet0_pt_NOSYS",
        "allLepBjet0_eta_NOSYS",
        "allLepBjet0_mass_NOSYS",
        "deltaPhi_Lep0_MET_NOSYS",
        "deltaR_Bjet0_ZCandidate0_NOSYS",
        "deltaPhi_ZCandidate0_allLepBjet0_NOSYS",
        "allLepJet_pt_NOSYS",
        "allLepJet_eta_NOSYS",
        "allJet_pt_NOSYS",
        "allJet_eta_NOSYS",
        "allBjet_pt_NOSYS",
        "allBjet_eta_NOSYS",
        "deltaPhi_Bjet0_ZCandidate0_NOSYS",
        "allLep_MET_mass_NOSYS",
        "allLep_MET_pt_NOSYS",
        "allLepBjet0_MET_pt_NOSYS",
        "dR_Z_l_min_NOSYS","dR_Z_l_max_NOSYS",
        "m_bl_min_NOSYS","m_bl_second_min_NOSYS",
        "dPhi_met_Z_NOSYS","dPhi_met_nonZ_lep0_NOSYS","dPhi_met_nonZ_lep1_NOSYS",
        "dPhi_met_bjet0_NOSYS","dPhi_met_bjet1_NOSYS",
        "mT_lep_met_min_NOSYS","mT_lep_met_max_NOSYS",
        "dR_nonZ_leps_NOSYS",
        "dR_Zlep_nonZlep_min_NOSYS",
        "m_bb_NOSYS","dR_bb_NOSYS",
        "dR_bl_min_NOSYS","dR_bl_secondmin_NOSYS"
    ]

    # Blank array for to conbine all individual datasets
    data = []
    
    for file in tWZ_files + ttZ_files + ZZ_files + other_files:
        file_path = Path(f"{input_dir}/{file}")
        if not file_path.exists():
            logging.warning(f"File not found: {file_path}")
            continue
        
        try:
            tree = uproot.open(file_path)["reco"]
            ak_array = tree.arrays(features, library="ak")
            df = ak.to_pandas(ak_array)
            df["label"] = (
                3 if file in tWZ_files
                else 2 if file in ttZ_files
                else 1 if file in ZZ_files
                else 0 
            )
            data.append(df)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue
    
    if not data:
        logging.error("No valid data files were loaded. Exiting.")
        return
    
    df = pd.concat(data, ignore_index=False).drop_duplicates()

    # Apply reweighting
    df["eventWeights"] = df["weight_total_NOSYS"]
    df = df.reset_index()

    if replot:
        for feature in (features):
            plot_histogram(df, feature, "eventWeights", output_dir)
    
    os.makedirs(f"{output_dir}/outputs", exist_ok=True)
    df.to_csv(f"{output_dir}/data_curated.csv", index=False)
    logging.info("Processing complete. Data saved.")

if __name__ == "__main__":
    args = parse_arguments()
    process_files(args.input_dir, args.output_dir, args.replot)
