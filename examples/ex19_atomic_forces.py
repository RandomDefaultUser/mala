import os

from ase.io import read
import mala
from mala import printout
import numpy as np

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

"""
ex12_run_predictions.py: Show how a prediction can be made using MALA.
Using nothing more then the trained network and atomic configurations, 
predictions can be made. 
"""


# Uses a network to make a prediction.
assert os.path.exists("be_model.zip"), "Be model missing, run ex01 first."
parameters, network, data_handler, predictor = mala.Predictor.\
    load_run("be_model")

####################
# PARAMETERS
# This may already have been done when training the network, but
# for predictions, the correct LDOS and descriptor parameters need to be known.
####################
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 11
parameters.targets.ldos_gridspacing_ev = 2.5
parameters.targets.ldos_gridoffset_ev = -5

parameters.descriptors.descriptor_type = "Bispectrum"
parameters.descriptors.bispectrum_twojmax = 10
parameters.descriptors.bispectrum_cutoff = 4.67637

# First, read atoms from some source, we'll simply use the simulation outputs
# of the training data.
atoms = read(os.path.join(data_path, "Be_snapshot3.out"))
ldos = predictor.predict_for_atoms(atoms)
ldos_calculator: mala.LDOS = predictor.target_calculator
ldos_calculator.read_from_array(ldos)
ldos_calculator.temperature = 4000
mala_forces = ldos_calculator.atomic_forces.copy()



delta_numerical = 1e-14
numerical_forces = []

for i in range(0, parameters.targets.ldos_gridsize):
    ldos_plus = ldos.copy()
    ldos_plus[0, i] += delta_numerical*0.5
    ldos_calculator.read_from_array(ldos_plus)
    derivative_plus = ldos_calculator.band_energy + \
                      ldos_calculator.entropy_contribution

    ldos_minus = ldos.copy()
    ldos_minus[0, i] -= delta_numerical*0.5
    ldos_calculator.read_from_array(ldos_minus)
    derivative_minus = ldos_calculator.band_energy + \
                       ldos_calculator.entropy_contribution

    numerical_forces.append((derivative_plus-derivative_minus) /
                            delta_numerical)

print(mala_forces[0, :])
print(mala_forces[2000, :])
print(mala_forces[4000, :])
print(np.array(numerical_forces))
print(mala_forces[4000, :]/np.array(numerical_forces))
