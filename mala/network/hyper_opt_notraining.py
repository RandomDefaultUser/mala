"""Hyperparameter optimizer working without training."""
import itertools

import optuna
import numpy as np

from mala.common.parallelizer import printout
from mala.network.hyper_opt_base import HyperOptBase
from mala.network.objective_no_training import ObjectiveNoTraining


class HyperOptNoTraining(HyperOptBase):
    """
    Hyperparameter optimizer that does not require training networks.

    Networks are analysed using the Jacobian.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this hyperparameter optimizer.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the data for the hyperparameter optimization.
    """

    def __init__(self, params, data):
        super(HyperOptNoTraining, self).__init__(params, data)
        self.objective = None
        self.trial_losses = None
        self.best_trial = None
        self.trial_list = None
        self.ignored_hyperparameters = ["learning_rate", "trainingtype",
                                        "mini_batch_size",
                                        "early_stopping_epochs",
                                        "learning_rate_patience",
                                        "learning_rate_decay"]

    def perform_study(self, trial_list=None):
        """
        Perform the study, i.e. the optimization.

        This is done by sampling a certain subset of network architectures.
        Currently it is mandatory to provide a trial_list, although it
        will be optional later on.

        Parameters
        ----------
        trial_list : list
            A list containing trials from either HyperOptOptuna or HyperOptOAT.
            HyperOptNoTraining does currently not have an algorithm to
            create network archtitectures of interestes by itself and insteads
            investigates those sampled by a different hyperparameter
            optimizer.
        """
        # The minibatch size can not vary in the analysis.
        # This check ensures that e.g. optuna results can be used.
        for idx, par in enumerate(self.params.hyperparameters.hlist):
            if par.name == "mini_batch_size":
                printout("Removing mini batch size from hyperparameter list, "
                         "because NASWOT is used.")
                self.params.hyperparameters.hlist.pop(idx)

        # Ideally, this type of HO is called with a list of trials for which
        # the parameter has to be identified.
        self.trial_list = trial_list
        if self.trial_list is None:
            printout("No trial list provided, one will be created using all "
                     "possible permutations of hyperparameters. "
                     "The following hyperparameters will be ignored:")
            printout(self.ignored_hyperparameters)
            self.trial_list = self.__all_combinations()

        # TODO: For now. Needs some refinements later.
        if isinstance(self.trial_list[0], optuna.trial.FrozenTrial) or \
           isinstance(self.trial_list[0], optuna.trial.FixedTrial):
            trial_type = "optuna"
        else:
            trial_type = "oat"
        self.objective = ObjectiveNoTraining(self.params, self.data_handler,
                                             trial_type)
        printout("Starting NASWOT hyperparameter optimization,",
                 len(self.trial_list), "trials will be performed.")

        self.trial_losses = []
        for idx, row in enumerate(self.trial_list):
            trial_loss = self.objective(row)
            self.trial_losses.append(trial_loss)

            # Output diagnostic information.
            best_trial = self.get_best_trial_results()
            printout("Trial number", idx,
                     "finished with:", self.trial_losses[idx],
                     ", best is trial", best_trial[0],
                     "with", best_trial[1])

        # Return the best loss value we could achieve.
        return self.get_best_trial_results()[1]

    def get_best_trial_results(self):
        """Get the best trial out of the list, including the value."""
        return [np.argmax(self.trial_losses), np.max(self.trial_losses)]

    def set_optimal_parameters(self):
        """
        Set the optimal parameters found in the present study.

        The parameters will be written to the parameter object with which the
        hyperparameter optimizer was created.
        """
        # Getting the best trial based on the test errors
        idx = self.trial_losses.index(max(self.trial_losses))
        self.best_trial = self.trial_list[idx]
        self.objective.parse_trial(self.best_trial)

    def __all_combinations(self):
        # First, remove all the hyperparameters we don't actuall need.
        indices_to_remove = []
        for idx, par in enumerate(self.params.hyperparameters.hlist):
            if par.name in self.ignored_hyperparameters:
                indices_to_remove.append(idx)
        for index in sorted(indices_to_remove, reverse=True):
            del self.params.hyperparameters.hlist[index]

        # Next, create a helper list to calculate all possible combinations.
        all_hyperparameters_choices = []
        for par in self.params.hyperparameters.hlist:
            all_hyperparameters_choices.append(par.choices)

        # Calculate all possible combinations.
        all_combinations = list(itertools.product(*all_hyperparameters_choices))

        # Now we use these combination to fill a list of FixedTrials.
        trial_list = []
        for combination in all_combinations:
            params_dict = {}
            for idx, value in enumerate(combination):
                params_dict[self.params.hyperparameters.hlist[idx].name] = value
            new_trial = optuna.trial.FixedTrial(params_dict)
            trial_list.append(new_trial)

        return trial_list
