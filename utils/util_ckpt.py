from __future__ import absolute_import, division, print_function
import os, shutil, datetime
import tensorflow as tf


def checkpoint_logger(global_step, model, optimizer, flag_resume, root_ckptlogs,
    subproject_name, exp_phase, comment, time_stamp, path_resume=None, 
    max_to_keep=3, config_path=None):
    """Make ckpt and manager objects, and restore the latest checkpoint if necessary.
    Args:
        global_step: A tf.Variable Tensor with dtype=tf.int32. tf.int64 didn't work...
        model: A tf.keras.Model object.
        optimizer: An optimizer object such as tf.optimizers.Adam(0.1)
        flag_resume: A boolean. Whether to resume training from the latest ckpt.
        root_ckptlogs: A string. Used for path to ckpts.
        subproject_name: A string. Used for path to ckpts.
        comment: A string. Used for path to ckpts.
        time_stamp: A string. Used for path to ckpts.
        path_resume: A string or None. The path to ckpt logs to be resumed. 
            path_resume is ignored if flag_resume=False.
        max_to_keep: An int. Set max_to_keep=0 or None to keep all the ckpts.
        config_path: A string, where config file is saved for reference.
    Returns:
        ckpt: tf.train.Checkpoint object.
        ckpt_manager: tf.train.CheckpointManager object.
    Remark:
        Path to checkpoint files is 
            'root_ckptlogs'/'subproject_name'_'exp_phase'/'comment'_'time_stamp'/ckptXXX
    """
    # Naming rule
    dir_ckptlogs = "{}/{}_{}/{}_{}".format(
        root_ckptlogs, subproject_name, exp_phase, comment, time_stamp)

    if not(config_path is None):
        dir_configs = dir_ckptlogs + "/configs"

        if not os.path.exists(dir_configs):
            os.makedirs(dir_configs)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]        
        shutil.copyfile(config_path, dir_configs + "/config_{}.yaml".format(now))

    # Create ckpt
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=model)

    # If resume
    if flag_resume:
        assert os.path.exists(path_resume), "Not exist: path_ckpt {}".format(
            path_resume)

        # Create ckpt and manager for restore
        ckpt_manager_restore = tf.train.CheckpointManager(
            ckpt, path_resume, max_to_keep=max_to_keep)

        # Restore the latest ckpt log.
        ckpt.restore(ckpt_manager_restore.latest_checkpoint)
        print("Restored from {}".format(ckpt_manager_restore.latest_checkpoint))        
    
    # Create manager
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, dir_ckptlogs, max_to_keep=max_to_keep)

    return ckpt, ckpt_manager
