"""Hyperparameter optimizer using orthogonal array tuning."""

from bisect import bisect
import itertools
import pickle

import numpy as np

try:
    import oapackage as oa
except ModuleNotFoundError:
    pass

from mala.network.hyper_opt import HyperOpt
from mala.network.objective_base import ObjectiveBase
from mala.network.hyperparameter_oat import HyperparameterOAT
from mala.common.parallelizer import printout, parallel_warn


class HyperOptGridsearch(HyperOpt):
    """Hyperparameter optimizer using Orthogonal Array Tuning.

    Based on https://link.springer.com/chapter/10.1007/978-3-030-36808-1_31.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, data, use_pkl_checkpoints=False):
        super(HyperOptGridsearch, self).__init__(
            params, data, use_pkl_checkpoints=use_pkl_checkpoints
        )
        self._objective = None
        self._optimal_params = None
        self._checkpoint_counter = 0

        # Related to the grid search.
        self._grid = None

        # Tracking the trial progress.
        self._sorted_num_choices = []
        self._current_trial = 0
        self._trial_losses = None

    def get_any_parameter_attribute(object, attribute):
        if "." in attribute:
            subparams, subattribute = attribute.split(".")
            return getattr(getattr(object, subparams), subattribute)
        else:
            return getattr(object, attribute)

    def add_hyperparameter(self, name="", choices=None, **kwargs):
        """
        Add hyperparameter.

        Hyperparameter list will automatically sorted w.r.t the number of
        choices.

        Parameters
        ----------
        opttype : string
            Datatype of the hyperparameter. Follows optuna's naming
            conventions, but currently only supports "categorical" (a list).
        """
        self.params.hyperparameters.hlist.append(
            HyperparameterOAT(
                opttype="categorical", name=name, choices=choices
            ),
        )

    def perform_study(self):
        """
        Perform the study, i.e. the optimization.

        Internally constructs an orthogonal array and performs trial NN
        trainings based on it.
        """
        number_choices = [
            range(h.num_choices) for h in self.params.hyperparameters.hlist
        ]
        for combination in itertools.product(*number_choices):
            print(combination)

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        self._objective.parse_trial_oat(self._optimal_params)

    @classmethod
    def resume_checkpoint(
        cls, checkpoint_name, no_data=False, use_pkl_checkpoints=False
    ):
        """
        Prepare resumption of hyperparameter optimization from a checkpoint.

        Please note that to actually resume the optimization,
        HyperOptOAT.perform_study() still has to be called.

        Parameters
        ----------
        checkpoint_name : string
            Name of the checkpoint from which the checkpoint is loaded.

        no_data : bool
            If True, the data won't actually be loaded into RAM or scaled.
            This can be useful for cases where a checkpoint is loaded
            for analysis purposes.

        use_pkl_checkpoints : bool
            If true, .pkl checkpoints will be loaded.

        Returns
        -------
        loaded_params : mala.common.parameters.Parameters
            The parameters saved in the checkpoint.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from the checkpoint.

        new_hyperopt : HyperOptOAT
            The hyperparameter optimizer reconstructed from the checkpoint.
        """
        loaded_params, new_datahandler, optimizer_name = (
            cls._resume_checkpoint(
                checkpoint_name,
                no_data=no_data,
                use_pkl_checkpoints=use_pkl_checkpoints,
            )
        )
        new_hyperopt = HyperOptOAT.load_from_file(
            loaded_params, optimizer_name, new_datahandler
        )

        return loaded_params, new_datahandler, new_hyperopt

    @classmethod
    def load_from_file(cls, params, file_path, data):
        """
        Load a hyperparameter optimizer from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the hyperparameter optimizer
            should be created Has to be compatible with data.

        file_path : string
            Path to the file from which the hyperparameter optimizer should
            be loaded.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.

        Returns
        -------
        loaded_hyperopt : HyperOptOAT
            The hyperparameter optimizer that was loaded from the file.
        """
        # First, load the checkpoint.
        with open(file_path, "rb") as handle:
            loaded_tracking_data = pickle.load(handle)
            loaded_hyperopt = HyperOptOAT(params, data)
            loaded_hyperopt._sorted_num_choices = loaded_tracking_data[
                "sorted_num_choices"
            ]
            loaded_hyperopt._current_trial = loaded_tracking_data[
                "current_trial"
            ]
            loaded_hyperopt._trial_losses = loaded_tracking_data[
                "trial_losses"
            ]
            loaded_hyperopt._importance = loaded_tracking_data["importance"]
            loaded_hyperopt._n_factors = loaded_tracking_data["n_factors"]
            loaded_hyperopt._factor_levels = loaded_tracking_data[
                "factor_levels"
            ]
            loaded_hyperopt._strength = loaded_tracking_data["strength"]
            loaded_hyperopt._N_runs = loaded_tracking_data["N_runs"]
            loaded_hyperopt.__OA = loaded_tracking_data["OA"]

        return loaded_hyperopt

    def __create_checkpointing(self, trial):
        """Create a checkpoint of optuna study, if necessary."""
        self._checkpoint_counter += 1
        need_to_checkpoint = False

        if (
            self._checkpoint_counter
            >= self.params.hyperparameters.checkpoints_each_trial
            and self.params.hyperparameters.checkpoints_each_trial > 0
        ):
            need_to_checkpoint = True
            printout(
                str(self.params.hyperparameters.checkpoints_each_trial)
                + " trials have passed, creating a "
                "checkpoint for hyperparameter "
                "optimization.",
                min_verbosity=1,
            )
        if (
            self.params.hyperparameters.checkpoints_each_trial < 0
            and np.argmin(self._trial_losses) == self._current_trial - 1
        ):
            need_to_checkpoint = True
            printout(
                "Best trial is "
                + str(self._current_trial - 1)
                + ", creating a "
                "checkpoint for it.",
                min_verbosity=1,
            )

        if need_to_checkpoint is True:
            # We need to create a checkpoint!
            self._checkpoint_counter = 0

            self._save_params_and_scaler()

            # The study only has to be saved if the no RDB storage is used.
            if self.params.hyperparameters.rdb_storage is None:
                hyperopt_name = (
                    self.params.hyperparameters.checkpoint_name
                    + "_hyperopt.pth"
                )

                study = {
                    "sorted_num_choices": self._sorted_num_choices,
                    "current_trial": self._current_trial,
                    "trial_losses": self._trial_losses,
                    "importance": self._importance,
                    "n_factors": self._n_factors,
                    "factor_levels": self._factor_levels,
                    "strength": self._strength,
                    "N_runs": self._N_runs,
                    "OA": self.__OA,
                }
                with open(hyperopt_name, "wb") as handle:
                    pickle.dump(study, handle, protocol=4)
