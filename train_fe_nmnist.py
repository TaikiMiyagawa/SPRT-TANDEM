from __future__ import absolute_import, division, print_function
import datetime, sys
import numpy as np
import tensorflow as tf

from datasets.data_processing import read_tfrecords_nosaic_mnist,\
    decode_nosaic_mnist, binarize_labels_nosaic_mnist,\
    normalize_images_nosaic_mnist, reshape_for_featext
from models.backbones_fe import ResNetModel, get_ressize_dependent_params
from models.optimizers import get_optimizer
from models.losses import get_loss_fe
from utils.misc import load_yaml, set_gpu_devices, fix_random_seed
from utils.util_tensorboard import TensorboardLogger
from utils.performance_metrics import list_metrics_to_list_bac,\
    dict_confmx_to_dict_metrics, logits_to_confmx, binary_confmx_to_bac
from utils.util_optuna import run_optuna, suggest_parameters_fe
from utils.util_ckpt import checkpoint_logger

# Load Params
config_path = "~/SPRT-TANDEM/configs/config_fe_nmnist.yaml"
config = load_yaml(config_path)

# GPU settings
set_gpu_devices(config["gpu"])

# Set Random Seeds (Optional)
fix_random_seed(flag_seed=config["flag_seed"], seed=config["seed"])


# Subfunctions
def tblog_writer_train(tblogger, losses, global_step, bac):
    tblogger.scalar_summary("train_loss/cross_entropy_loss", losses[1], int(global_step))
    tblogger.scalar_summary("train_metric/balanced_accuracy", bac, int(global_step))


def tblog_writer_valid(tblogger, losses_valid, global_step, bac_valid):
    tblogger.scalar_summary("valid_loss/cross_entropy_loss", losses_valid[1], int(global_step))
    tblogger.scalar_summary("valid_metric/balanced_accuracy", bac_valid, int(global_step))


def validation_loop(parsed_image_dataset_valid, model, num_validdata, batch_size, feat_dims):
    # Validation loop
    for iter_bv, feats_valid in enumerate(parsed_image_dataset_valid):
        cnt = iter_bv + 1

        # Decode features and binarize classification labels
        x_batch_valid, y_batch_valid = decode_nosaic_mnist(feats_valid) 
        x_batch_valid, y_batch_valid = reshape_for_featext(
            x_batch_valid, y_batch_valid, feat_dims)
        x_batch_valid = normalize_images_nosaic_mnist(x_batch_valid)
        y_batch_valid = binarize_labels_nosaic_mnist(y_batch_valid)

        # Calc loss
        _, losses_valid_tmp, logits_valid, _ = get_loss_fe(
            model, 
            x_batch_valid, 
            y_batch_valid,
            flag_wd=False,
            training=False, 
            calc_grad=False,
            param_wd=None)

        # Calc confusion matrix of multiplets for every frame
        confmx_valid_tmp = logits_to_confmx(logits_valid, y_batch_valid)

        if iter_bv == 0:
            confmx_valid = confmx_valid_tmp
            losses_valid = losses_valid_tmp

        else:
            confmx_valid += confmx_valid_tmp
            for iter_idx in range(len(losses_valid_tmp)):
                losses_valid[iter_idx] += losses_valid_tmp[iter_idx]

        # Verbose
        if ((iter_bv+1)%10 == 0) or (iter_bv == 0):
            sys.stdout.write(
                "\rValidation Iter: {:3d}/{:3d}".format(iter_bv+1, (num_validdata//batch_size) + 1))
            sys.stdout.flush()

    print()

    # Normalization and calc metrics from confmx
    for iter_idx in range(len(losses_valid)):
        losses_valid[iter_idx] /= cnt

    bac_valid = binary_confmx_to_bac(confmx_valid)
    print(confmx_valid)
    print("Balanced Accuracy: {:7.5f}".format(bac_valid))

    return losses_valid, bac_valid


# Main Function
def objective(trial):
    # Timestamp
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

    # Suggest parameters if necessary
    ####################################
    assert (config["exp_phase"] == "tuning") or\
         (config["exp_phase"] == "stat") or (config["exp_phase"] == "try")
    if config["exp_phase"] == "tuning":
        list_suggest = suggest_parameters_fe(
            trial, 
            list_lr=config["list_lr"], 
            list_bs=config["list_bs"], 
            list_opt=config["list_opt"], 
            list_do=config["list_do"], 
            list_wd=config["list_wd"])

        learning_rate = list_suggest[0]
        batch_size = list_suggest[1]
        name_optimizer = list_suggest[2]
        dropout = list_suggest[3]
        weight_decay = list_suggest[4]

        config["learning_rates"] = [learning_rate, learning_rate*0.1, learning_rate*0.01]
        config["batch_size"] = batch_size
        config["name_optimizer"] = name_optimizer
        config["dropout"] = dropout
        config["weight_decay"] = weight_decay

    # Load data
    ##################################
    # Reed tfr
    parsed_image_dataset_train, parsed_image_dataset_valid, parsed_image_dataset_test =\
        read_tfrecords_nosaic_mnist(
            record_file_train=config["tfr_train"], 
            record_file_test=config["tfr_test"], 
            batch_size=config["batch_size"], 
            shuffle_buffer_size=2000)
        
    # Model
    ######################################
    dict_resparams = get_ressize_dependent_params(
        config["resnet_version"], config["resnet_size"])

    model = ResNetModel(
        resnet_size=config["resnet_size"],
        bottleneck=dict_resparams["bottleneck"],
        num_classes=config["nb_cls"],
        kernel_size=dict_resparams["kernel_size"],
        conv_stride=dict_resparams["conv_stride"],
        first_pool_size=dict_resparams["first_pool_size"],
        first_pool_stride=dict_resparams["first_pool_stride"],
        block_sizes=dict_resparams["block_sizes"],
        block_strides=dict_resparams["block_strides"],
        final_size=config["final_size"],
        resnet_version=config["resnet_version"],
        data_format='channels_last',
        dtype=tf.float32
    )


    # Get optimizer
    optimizer, flag_wd_in_loss = get_optimizer(
        learning_rates=config["learning_rates"], 
        decay_steps=config["decay_steps"], 
        name_optimizer=config["name_optimizer"], 
        flag_wd=config["flag_wd"], 
        weight_decay=config["weight_decay"])        


    # Tensorboard and checkpoints
    ####################################
    # Define global step
    global_step = tf.Variable(0, name="global_step", dtype=tf.int32)

    # Checkpoint
    _, ckpt_manager = checkpoint_logger(
        global_step, 
        model, 
        optimizer, 
        config["flag_resume"], 
        config["root_ckptlogs"], 
        config["subproject_name"], 
        config["exp_phase"],
        config["comment"], 
        now, 
        config["path_resume"], 
        config["max_to_keep"],
        config_path)

    # Tensorboard
    tblogger = TensorboardLogger(
        root_tblogs=config["root_tblogs"], 
        subproject_name=config["subproject_name"], 
        exp_phase=config["exp_phase"], 
        comment=config["comment"], 
        time_stamp=now)


    # Training
    ####################################
    # Start training
    with tblogger.writer.as_default():
        # Initialization
        best = 0.

        # Training and validation
        for epoch in range(config["nb_epochs"]):
            # Training loop
            for iter_b, feats in enumerate(parsed_image_dataset_train):
                # Decode features, normalize images, and binarize classification labels
                x_batch, y_batch = decode_nosaic_mnist(feats)
                x_batch, y_batch = reshape_for_featext(x_batch, y_batch, config["feat_dims"])
                x_batch = normalize_images_nosaic_mnist(x_batch)
                y_batch = binarize_labels_nosaic_mnist(y_batch)                

                # Show summary of model
                if (epoch == 0) and (iter_b == 0):
                    model.build(input_shape=x_batch.shape)
                    model.summary() 

                # Calc loss and grad, and backpropagation
                grads, losses, logits, _ = get_loss_fe(
                    model, 
                    x_batch, 
                    y_batch, 
                    training=True, 
                    param_wd=config["weight_decay"], 
                    flag_wd=flag_wd_in_loss,
                    calc_grad=True
                    )
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                global_step.assign_add(1)
                
                #Verbose
                if tf.equal(global_step % config["train_display_step"], 0) or tf.equal(global_step, 1):
                    print("Global Step={:7d} Epoch={:4d}/{:4d} Iter={:5d}/{:5d}: xent loss={:7.5f}"
                        .format(
                            int(global_step), 
                            epoch+1, 
                            config["nb_epochs"], 
                            iter_b+1, 
                            (config["num_traindata"]//config["batch_size"])+1, 
                            losses[1].numpy()))

                    # Confusion matrix of multiplets for every frame
                    confmx = logits_to_confmx(logits, y_batch)
                    print(confmx)
                    bac = binary_confmx_to_bac(confmx)
                    print("Balanced Accuracy: {:7.5f}".format(bac))

                    # Tensorboard
                    tblog_writer_train(
                        tblogger, 
                        losses, 
                        global_step, 
                        bac)

                # Validation
                if tf.equal(global_step % config["valid_step"], 0) or tf.equal(global_step, 1):
                    losses_valid, bac_valid = \
                        validation_loop(
                            parsed_image_dataset_valid, 
                            model, 
                            config["num_validdata"], 
                            config["batch_size"],
                            config["feat_dims"])

                    # Tensorboard for validation
                    tblog_writer_valid(
                        tblogger, 
                        losses_valid, 
                        global_step, 
                        bac_valid)           

                    # For exp_phase="tuning", optuna
                    if best < bac_valid:
                        best = bac_valid
                    
                        # Save checkpoint
                        ckpt_manager._checkpoint_prefix = \
                            ckpt_manager._checkpoint_prefix[:ckpt_manager._checkpoint_prefix.rfind("/") + 1] + \
                            "ckpt_step{}_mbac{:.5f}".format(int(global_step), best)
                        save_path_prefix = ckpt_manager.save()
                        print("Saved checkpoint for step {}: {}".format(
                            int(global_step), save_path_prefix))

                        # Test
                        ############################
                        if config["exp_phase"] == "stat":
                            # Test loop
                            _, bac_test = validation_loop(
                                parsed_image_dataset_test, 
                                model, 
                                config["num_testdata"], 
                                config["batch_size"],
                                config["feat_dims"])
     
    # Final processes
    ###############################################
    if config["exp_phase"] == "stat":
        # Save best values to .db file 
        trial.set_user_attr("bac", float(bac_test))

    # Return best valid result for optuna
    return 1 - best


if __name__ == '__main__':
    run_optuna(
        config["root_dblogs"], 
        config["subproject_name"], 
        config["exp_phase"], 
        objective, 
        config["nb_trials"])

