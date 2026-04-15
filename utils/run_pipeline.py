import subprocess

# Curate data
# curate_data = [
#     "python", "./scripts/dataCuration_fourClassModel.py",
#     "--input_dir", "./inputs/",
#     "--output_dir", "./outputs/",
#     "--replot"
# ]

# subprocess.run(curate_data, check=True)

# Train & Test the model
train_model = [
    "python", "./scripts/train_model.py",
    "--input_file", "./outputs/data_curated.csv",
    "--output_dir", "./outputs/trained/",
    "--hyperparameter_file", "./utils/hyperparameters.json"
]

subprocess.run(train_model, check=True)