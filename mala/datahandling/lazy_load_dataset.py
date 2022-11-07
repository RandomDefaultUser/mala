"""DataSet for lazy-loading."""
import os

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class.
    pass
import numpy as np
import torch
from torch.utils.data import Dataset

from mala.common.parallelizer import barrier
from mala.datahandling.snapshot import Snapshot


class LazyLoadDataset(torch.utils.data.Dataset):
    """
    DataSet class for lazy loading.

    Only loads snapshots in the memory that are currently being processed.
    Uses a "caching" approach of keeping the last used snapshot in memory,
    until values from a new ones are used. Therefore, shuffling at DataSampler
    / DataLoader level is discouraged to the point that it was disabled.
    Instead, we mix the snapshot load order here ot have some sort of mixing
    at all.

    Parameters
    ----------
    input_dimension : int
        Dimension of an input vector.

    output_dimension : int
        Dimension of an output vector.

    input_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the input data.

    output_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the output data.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data.

    target_calculator : mala.targets.target.Target or derivative
        Used to do unit conversion on output data.

    grid_dimensions : list
        Dimensions of the grid (x,y,z).

    grid_size : int
        Size of the grid (x*y*z), i.e. the number of datapoints per
        snapshot.

    use_horovod : bool
        If true, it is assumed that horovod is used.

    input_requires_grad : bool
        If True, then the gradient is stored for the inputs.
    """

    def __init__(self, input_dimension, output_dimension, input_data_scaler,
                 output_data_scaler, descriptor_calculator,
                 target_calculator, grid_dimensions, grid_size, use_horovod,
                 input_requires_grad=False,
                 grid_type="1d", data_splitting_3d=[0, 0, 0]):
        self.snapshot_list = []
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_data_scaler = input_data_scaler
        self.output_data_scaler = output_data_scaler
        self.descriptor_calculator = descriptor_calculator
        self.target_calculator = target_calculator
        self.grid_dimensions = grid_dimensions
        self.grid_size = grid_size
        self.number_of_snapshots = 0
        self.total_size = 0
        self.descriptors_contain_xyz = self.descriptor_calculator.\
            descriptors_contain_xyz
        self.currently_loaded_file = None
        self.input_data = np.empty(0)
        self.output_data = np.empty(0)
        self.use_horovod = use_horovod
        self.return_outputs_directly = False
        self.input_requires_grad = input_requires_grad
        self.grid_type = grid_type

        # For batched 3D data.
        self.x_fractions = None
        self.y_fractions = None
        self.z_fractions = None
        self.data_splitting_3d = data_splitting_3d

        # Default: Volume is not split.
        self.number_of_fractional_volumes = 1

        self.x_fractions = self.grid_dimensions[0]
        self.y_fractions = self.grid_dimensions[1]
        self.z_fractions = self.grid_dimensions[2]

        if self.data_splitting_3d[0] != 0 and \
                self.data_splitting_3d[1] != 0 and \
                self.data_splitting_3d[2] != 0:
            self.x_fractions = \
                int(self.grid_dimensions[0] / \
                    self.data_splitting_3d[0])
            self.y_fractions = \
                int(self.grid_dimensions[1] / \
                    self.data_splitting_3d[1])
            self.z_fractions = \
                int(self.grid_dimensions[2] / \
                    self.data_splitting_3d[2])
            self.number_of_fractional_volumes = \
                self.x_fractions * self.y_fractions * \
                self.z_fractions

    @property
    def return_outputs_directly(self):
        """
        Control whether outputs are actually transformed.

        Has to be False for training. In the testing case,
        Numerical errors are smaller if set to True.
        """
        return self._return_outputs_directly

    @return_outputs_directly.setter
    def return_outputs_directly(self, value):
        self._return_outputs_directly = value

    def add_snapshot_to_dataset(self, snapshot: Snapshot):
        """
        Add a snapshot to a DataSet.

        Afterwards, the DataSet can and will load this snapshot as needed.

        Parameters
        ----------
        snapshot : mala.datahandling.snapshot.Snapshot
            Snapshot that is to be added to this DataSet.

        """
        self.snapshot_list.append(snapshot)
        self.number_of_snapshots += 1
        self.total_size = self.number_of_snapshots*self.grid_size

    def mix_datasets(self):
        """
        Mix the order of the snapshots.

        With this, there can be some variance between runs.
        """
        used_perm = torch.randperm(self.number_of_snapshots)
        barrier()
        if self.use_horovod:
            used_perm = hvd.broadcast(used_perm, 0)
        self.snapshot_list = [self.snapshot_list[i] for i in used_perm]
        self.get_new_data(0)

    def get_new_data(self, file_index):
        """
        Read a new snapshot into RAM.

        Parameters
        ----------
        file_index : i
            File to be read.
        """
        # Load the data into RAM.
        self.input_data = \
            np.load(os.path.join(
                    self.snapshot_list[file_index].input_npy_directory,
                    self.snapshot_list[file_index].input_npy_file))
        self.output_data = \
            np.load(os.path.join(
                    self.snapshot_list[file_index].output_npy_directory,
                    self.snapshot_list[file_index].output_npy_file))

        # Transform the data.
        if self.descriptors_contain_xyz:
            self.input_data = self.input_data[:, :, :, 3:]
        if self.grid_type == "3d":
            if self.data_splitting_3d[0] != 0 and \
                    self.data_splitting_3d[1] != 0 and \
                    self.data_splitting_3d[2] != 0:

                # TODO: Make efficient.
                tmp = []
                for x in range(0, self.x_fractions):
                    for y in range(0, self.y_fractions):
                        for z in range(0, self.z_fractions):
                            tmp_tmp = self.input_data[x *
                                          self.data_splitting_3d[
                                              0]:(x + 1) *
                                                 self.data_splitting_3d[
                                                     0],
                                      y *
                                      self.data_splitting_3d[
                                          1]:(y + 1) *
                                             self.data_splitting_3d[
                                                 1],
                                      z *
                                      self.data_splitting_3d[
                                          2]:(z + 1) *
                                             self.data_splitting_3d[
                                                 2],
                                      :]

                            tmp_tmp = np.array(tmp_tmp).transpose(
                                [3, 0, 1, 2])
                            tmp_tmp *= self.descriptor_calculator. \
                                convert_units(1, self.snapshot_list[file_index].input_units)
                            tmp.append(tmp_tmp)
                self.input_data = np.array(tmp)
            else:
                self.input_data = np.array(self.input_data).transpose([3, 0, 1, 2])
                self.input_data *= self.descriptor_calculator. \
                    convert_units(1, self.snapshot_list[file_index].input_units)
        else:
            self.input_data *= \
                self.descriptor_calculator.\
                convert_units(1, self.snapshot_list[file_index].input_units)
            self.input_data = \
                self.input_data.reshape([self.grid_size, self.input_dimension])

        self.input_data = self.input_data.astype(np.float32)
        self.input_data = torch.from_numpy(self.input_data).float()
        self.input_data = self.input_data_scaler.transform(self.input_data)
        self.input_data.requires_grad = self.input_requires_grad

        if self.grid_type == "3d":
            if self.data_splitting_3d[0] != 0 and \
                    self.data_splitting_3d[1] != 0 and \
                    self.data_splitting_3d[2] != 0:

                # TODO: Make efficient.
                tmp = []
                for x in range(0, self.x_fractions):
                    for y in range(0, self.y_fractions):
                        for z in range(0, self.z_fractions):
                            tmp_tmp = self.output_data[
                                      x * self.data_splitting_3d[
                                          0]:(x + 1) *
                                             self.data_splitting_3d[
                                                 0],
                                      y * self.data_splitting_3d[
                                          1]:(y + 1) *
                                             self.data_splitting_3d[
                                                 1],
                                      z * self.data_splitting_3d[
                                          2]:(z + 1) *
                                             self.data_splitting_3d[
                                                 2],
                                      :]

                            tmp_tmp = np.array(tmp_tmp).transpose([3, 0, 1, 2])
                            tmp_tmp *= self.target_calculator. \
                                convert_units(1, self.snapshot_list[file_index].output_units)
                            tmp.append(tmp_tmp)
                self.output_data = np.array(tmp)
            else:
                self.output_data = self.output_data.transpose([3, 0, 1, 2])
                self.output_data *= self.target_calculator. \
                    convert_units(1, self.snapshot_list[file_index].output_units)
        else:
            self.output_data = \
                self.output_data.reshape(
                    [self.grid_size, self.output_dimension])
            self.output_data *= \
                self.target_calculator. \
                    convert_units(1,
                                  self.snapshot_list[file_index].output_units)

        if self.return_outputs_directly is False:
            self.output_data = np.array(self.output_data)
            self.output_data = self.output_data.astype(np.float32)
            self.output_data = torch.from_numpy(self.output_data).float()
            self.output_data = \
                self.output_data_scaler.transform(self.output_data)

        # Save which data we have currently loaded.
        self.currently_loaded_file = file_index

    def __getitem__(self, idx):
        """
        Get an item of the DataSet.

        Parameters
        ----------
        idx : int
            Requested index. NOTE: Slices over multiple files
            are currently NOT supported.

        Returns
        -------
        inputs, outputs : torch.Tensor
            The requested inputs and outputs
        """
        # Get item can be called with an int or a slice.
        if isinstance(idx, int):
            if self.grid_type == "1d":
                file_index = idx // self.grid_size
                index_in_file = idx % self.grid_size
            else:
                file_index = idx // self.number_of_fractional_volumes
                index_in_file = idx % self.number_of_fractional_volumes

            # Find out if new data is needed.
            if file_index != self.currently_loaded_file:
                self.get_new_data(file_index)
            return self.input_data[index_in_file], \
                self.output_data[index_in_file]

        elif isinstance(idx, slice):
            # If a slice is requested, we have to find out if t spans files.
            if self.grid_type == "1d":
                file_index_start = idx.start // self.grid_size
                index_in_file_start = idx.start % self.grid_size
                file_index_stop = idx.stop // self.grid_size
                index_in_file_stop = idx.stop % self.grid_size
            else:
                file_index_start = idx.start // self.number_of_fractional_volumes
                index_in_file_start = idx.start % self.number_of_fractional_volumes
                file_index_stop = idx.stop // self.number_of_fractional_volumes
                index_in_file_stop = idx.stop % self.number_of_fractional_volumes

            # If it does, we cannot deliver.
            # Take care though, if a full snapshot is requested,
            # the stop index will point to the wrong file.
            if file_index_start != file_index_stop:
                if index_in_file_stop == 0:
                    if self.grid_type == "1d":
                        index_in_file_stop = self.grid_size
                    else:
                        index_in_file_stop = self.number_of_fractional_volumes
                else:
                    raise Exception("Lazy loading currently only supports "
                                    "slices in one file. "
                                    "You have requested a slice over two "
                                    "files.")

            # Find out if new data is needed.
            file_index = file_index_start
            if file_index != self.currently_loaded_file:
                self.get_new_data(file_index)
            return self.input_data[index_in_file_start:index_in_file_stop], \
                self.output_data[index_in_file_start:index_in_file_stop]

    def __len__(self):
        """
        Get the length of the DataSet.

        Returns
        -------
        length : int
            Number of data points in DataSet.
        """
        return self.total_size
