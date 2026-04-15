# tWZ Measurement - DNN energy dependent study

This repo will be used to study the energy dependence of the DNN within the tWZ measurement. The proton-proton collision data collected via the LHC by the ATLAS detector ran with a center-of-mass energy of 13 TeV for Run 2 and 13.6 TeV for Run 3. We are specifically looking into measuring the tWZ process from the 4-lepton decay channel, which is clean in signal but statistically limited. Therefore, we deploy a DNN to separate signal and background to enhance the measurement. The aim of this study is to test the stability of the DNN across collision energies. If the DNN is stable, we can train the model using both Run 2 and Run 3 together, which increases the statistics and produce more accurate classifications.

To clone the repo:
```
git clone https://github.com/drkevinbarends/twz_ml_energy_study.git
```

Create a branch to work in:
```
git checkout -b my-branch-name
```

To setup the environment:
```
source utils/setup.sh
```

To run the full pipeline - curate the data and train the model:
```
python utils/run_pipeline.py
```