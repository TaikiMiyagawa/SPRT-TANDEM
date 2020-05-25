import optuna
import sys, os

def run_optuna(root_dblogs, subproject_name, exp_phase, objective, nb_trials):
    """ Load or create study and optimize objective.
    Args:
        root_dblogs: A str. The root directory for .db files, e.g., "/data/t-miyagawa/PROJECT_NAME/DATASET/dblogs"
        sugproject_name: A string. Used for study name and storage name.
        exp_phase: A string. Used for study name and storage name.
    """
    # Paths
    study_name = subproject_name + "_" + exp_phase
    storage_name = "sqlite:///" + root_dblogs + "/" + study_name + ".db"
    if not os.path.exists(root_dblogs):
        os.makedirs(root_dblogs)

    # Load or create study, and start optimization
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=nb_trials)


def suggest_parameters(trial, list_lr, list_bs, list_opt, list_do, list_wi):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_do: dropout
        list_wi: number of hidden units in LSTM
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
    """
    # load yaml interprrets 1e-2 as string
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    dropout = trial.suggest_categorical('dropout', list_do)
    width = trial.suggest_categorical('width', list_wi)

    return learning_rate, batch_size, name_optimizer, dropout, width


def suggest_parameters_fe(trial, list_lr, list_bs, list_opt, list_do, list_wd):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_do: dropout
        list_wd: weight decay 
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
        ...
    """
    # load yaml interprrets 1e-2 as string
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    dropout = trial.suggest_categorical('dropout', list_do)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)

    return learning_rate, batch_size, name_optimizer, dropout, weight_decay

