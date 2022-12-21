import mala
import numpy as np
import matplotlib.pyplot as plt

params = mala.Parameters()
params.targets.target_type = "LDOS"
params.targets.ldos_gridsize = 11
params.targets.ldos_gridspacing_ev = 2.5
params.targets.ldos_gridoffset_ev = -5
params.data.data_dimensions = "3d"
params.data.output_rescaling_type = "normal"
params.network.number_of_input_channels = 1
params.network.number_of_output_channels = 22
params.network.layer_sizes = [22, 11]
params.network.kernel_size = 21
params.descriptors.descriptors_contain_xyz = False
params.network.nn_type = "locality-cnn"
params.running.max_number_epochs = 500
params.running.learning_rate = 2
# params.data.data_splitting_3d = [3, 3, 3]

data_handler = mala.DataHandler(params)

inpath = "/home/fiedlerl/data/mala_data_repo/Be2/" \
         "densities_gp/inputs_gaussians/"
outpath = "/home/fiedlerl/data/mala_data_repo/Be2/training_data/"

data_handler.add_snapshot("gaussians_optimized1.in.npy", inpath,
                          "Be_snapshot1.out.npy", outpath,
                          add_snapshot_as="tr", output_units="1/(eV*Bohr^3)")
data_handler.add_snapshot("gaussians_optimized2.in.npy", inpath,
                          "Be_snapshot2.out.npy", outpath,
                          add_snapshot_as="va", output_units="1/(eV*Bohr^3)")
data_handler.prepare_data()

network = mala.Network(params)
trainer = mala.Trainer(params, network, data_handler)
trainer.train_network()

params.data.use_lazy_loading = True
data_handler.clear_data()
data_handler.add_snapshot("gaussians_optimized1.in.npy", inpath,
                          "Be_snapshot1.out.npy", outpath,
                          add_snapshot_as="te", output_units="1/(eV*Bohr^3)")
data_handler.prepare_data(reparametrize_scaler=False)

tester = mala.Tester(params, network, data_handler)
actual_ldos, predicted_ldos = tester.test_snapshot(0)
ldos_calculator: mala.LDOS
ldos_calculator = data_handler.target_calculator
ldos_calculator.read_additional_calculation_data("qe.out", outpath+"Be_snapshot1.out")

ldos_calculator.read_from_array(actual_ldos)
density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
density_calculator.write_as_cube("actual.cube", density_calculator.density)

ldos_calculator.read_from_array(predicted_ldos)
density_calculator = mala.Density.from_ldos_calculator(ldos_calculator)
density_calculator.write_as_cube("cnn.cube", density_calculator.density)

gaussians = np.load(inpath+"gaussians_optimized1.in.npy").reshape([18*18*27, 1])
density_calculator.write_as_cube("gaussians.cube", gaussians)


# print(ldos_calculator.number_of_electrons, ldos_calculator.band_energy)
# actual_dos = ldos_calculator.density_of_states.copy()
# energy_grid = ldos_calculator.energy_grid
# ldos_calculator.read_from_array(predicted_ldos)
# print(ldos_calculator.number_of_electrons, ldos_calculator.band_energy)
# predicted_dos = ldos_calculator.density_of_states.copy()
# plt.plot(energy_grid, actual_dos, label="Actual DOS")
# plt.plot(energy_grid, predicted_dos, label="Predicted DOS")
# plt.legend()
# plt.show()



