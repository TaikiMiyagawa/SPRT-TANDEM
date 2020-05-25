from __future__ import absolute_import, division, print_function
import datetime, sys
import numpy as np
import tensorflow as tf

from datasets.data_processing import read_tfrecords_nosaic_mnist,\
    decode_feat
from models.backbones_ti import LSTMModelLite
from models.optimizers import get_optimizer
from models.losses import get_gradient_lstm, get_loss_lstm
from utils.misc import load_yaml, set_gpu_devices, fix_random_seed
from utils.util_tensorboard import TensorboardLogger
from utils.performance_metrics import multiplet_sequential_confmx,\
    binary_llr_sequential_confmx, binary_truncated_sprt, seqconfmx_to_list_metrics,\
    list_metrics_to_list_bac, dict_confmx_to_dict_metrics, run_truncated_sprt
from utils.util_optuna import run_optuna, suggest_parameters_fe
from utils.util_ckpt import checkpoint_logger

# Load Params
config_path = "~/SPRT-TANDEM/configs/config_ti_nmnist.yaml"
config = load_yaml(config_path)

# GPU settings
set_gpu_devices(config["gpu"])

# Set Random Seeds (Optional)
fix_random_seed(flag_seed=config["flag_seed"], seed=config["seed"])


# Subfunctions
def tblog_writer_train(tblogger, losses, global_step, dict_metrics_sprt,
    dict_mean_hittimes, dict_truncate_rates, list_metrics_mult, list_metrics_llr):
    tblogger.scalar_summary("train_loss/sum_loss", losses[1]+losses[2], int(global_step))
    tblogger.scalar_summary("train_loss/multiplet_loss", losses[1], int(global_step))
    tblogger.scalar_summary("train_loss/llr_loss", losses[2], int(global_step))

    for key in dict_metrics_sprt.keys():
        metrics = dict_metrics_sprt[key]
        mean_hittime = dict_mean_hittimes[key]
        truncate_rate = dict_truncate_rates[key]
        tblogger.scalar_summary(
            "train_metric_sprt/mean_hitting_time_{}".format(key), mean_hittime, int(global_step))
        tblogger.scalar_summary(
            "train_metric_sprt/balanced_accuracy_{}".format(key), metrics["BAC"][0], int(global_step))
        tblogger.scalar_summary(
            "train_metric_sprt/truncate_rate_{}".format(key), truncate_rate, int(global_step))
        tblogger.scalar_summary(
            "train_metric_sprt_detail/sensitivity_{}".format(key), metrics["SNS"][0], int(global_step))
        tblogger.scalar_summary(
            "train_metric_sprt_detail/specificity_{}".format(key), metrics["SNS"][1], int(global_step))

    # tblogger.scalar_summary(
    #     "train_metric_multiplet/sensitivity_frame020", list_metrics_mult[-1]["SNS"][0], int(global_step))
    # tblogger.scalar_summary(
    #     "train_metric_multiplet/specificity/frame020", list_metrics_mult[-1]["SNS"][1], int(global_step))
    tblogger.scalar_summary(
        "train_metric_multiplet/balance_accuracy_frame001", list_metrics_mult[0]["BAC"][0], int(global_step))
    tblogger.scalar_summary(
        "train_metric_multiplet/balance_accuracy_frame010", list_metrics_mult[9]["BAC"][0], int(global_step))
    tblogger.scalar_summary(
        "train_metric_multiplet/balance_accuracy_frame020", list_metrics_mult[-1]["BAC"][0], int(global_step))

    # tblogger.scalar_summary(
    #     "train_metric_llr/sensitivity_frame020", list_metrics_llr[-1]["SNS"][0], int(global_step))              
    # tblogger.scalar_summary(
    #    "train_metric_llr/specificity_frame020", list_metrics_llr[-1]["SNS"][1], int(global_step))              
    tblogger.scalar_summary(
        "train_metric_llr/balanced_accuracy_frame001", list_metrics_llr[0]["BAC"][0], int(global_step)) 
    tblogger.scalar_summary(
        "train_metric_llr/balanced_accuracy_frame010", list_metrics_llr[9]["BAC"][0], int(global_step)) 
    tblogger.scalar_summary(
        "train_metric_llr/balanced_accuracy_frame020", list_metrics_llr[-1]["BAC"][0], int(global_step)) 


def tblog_writer_valid(tblogger, losses_valid, global_step, mean_bac_valid, 
    dict_metrics_sprt_valid, dict_mean_hittimes_valid, dict_truncate_rates_valid, 
    list_metrics_mult_valid, list_metrics_llr_valid):
    tblogger.scalar_summary(
        "valid_loss/sum_loss", losses_valid[1] + losses_valid[2], int(global_step))
    tblogger.scalar_summary(
        "valid_loss/multiplet_loss", losses_valid[1], int(global_step))
    tblogger.scalar_summary(
        "valid_loss/llr_loss", losses_valid[2], int(global_step))      
    tblogger.scalar_summary(
        "valid_metric_llr/mean_balanced_accuracy", mean_bac_valid, int(global_step))

    for key in dict_metrics_sprt_valid.keys():
        metrics_valid = dict_metrics_sprt_valid[key]
        mean_hittime_valid = dict_mean_hittimes_valid[key]
        truncate_rate_valid = dict_truncate_rates_valid[key]
        tblogger.scalar_summary(
            "valid_metric_sprt/mean_hitting_time_{}".format(key), mean_hittime_valid, int(global_step))
        tblogger.scalar_summary(
            "valid_metric_sprt/balanced_accuracy_{}".format(key), metrics_valid["BAC"][0], int(global_step))
        tblogger.scalar_summary(
            "valid_metric_sprt/truncate_rate_{}".format(key), truncate_rate_valid, int(global_step))
        tblogger.scalar_summary(
            "valid_metric_sprt_detail/sensitivity_{}".format(key), metrics_valid["SNS"][0], int(global_step))
        tblogger.scalar_summary(
            "valid_metric_sprt_detail/specificity_{}".format(key), metrics_valid["SNS"][1], int(global_step))

    # tblogger.scalar_summary(
    #     "valid_metric_multiplet/sensitivity_frame020", 
    #     list_metrics_mult_valid[-1]["SNS"][0], int(global_step))
    # tblogger.scalar_summary(
    #     "valid_metric_multiplet/specificity_frame020", 
    #     list_metrics_mult_valid[-1]["SNS"][1], int(global_step))
    tblogger.scalar_summary(
        "valid_metric_multiplet/balance_accuracy_frame001", 
        list_metrics_mult_valid[0]["BAC"][0], int(global_step))
    tblogger.scalar_summary(
        "valid_metric_multiplet/balance_accuracy_frame010", 
        list_metrics_mult_valid[9]["BAC"][0], int(global_step))
    tblogger.scalar_summary(
        "valid_metric_multiplet/balance_accuracy_frame020", 
        list_metrics_mult_valid[-1]["BAC"][0], int(global_step))

    # tblogger.scalar_summary(
    #     "valid_metric_llr/sensitivity_frame020", 
    #     list_metrics_llr_valid[-1]["SNS"][0], int(global_step))              
    # tblogger.scalar_summary(
    #     "valid_metric_llr/specificity_frame020", 
    #     list_metrics_llr_valid[-1]["SNS"][1], int(global_step))    
    tblogger.scalar_summary(
        "valid_metric_llr/balanced_accuracy_frame001", 
        list_metrics_llr_valid[0]["BAC"][0], int(global_step))   
    tblogger.scalar_summary(
        "valid_metric_llr/balanced_accuracy_frame010", 
        list_metrics_llr_valid[9]["BAC"][0], int(global_step))            
    tblogger.scalar_summary(
        "valid_metric_llr/balanced_accuracy_frame020", 
        list_metrics_llr_valid[-1]["BAC"][0], int(global_step))


def validation_loop(parsed_image_dataset_valid, model, order_sprt, duration,
    list_alpha, list_beta, num_validdata, batch_size, feat_dim):
    # Validation loop
    for iter_bv, feats_valid in enumerate(parsed_image_dataset_valid):
        cnt = iter_bv + 1

        # Decode features and binarize classification labels
        x_batch_valid, y_batch_valid = decode_feat(feats_valid, 
            config["duration"], config["feat_dim"], 
            dtype_feat=tf.float32, dtype_label=tf.int32) 

        # Calc loss, confmx, and mean hitting time 
        if iter_bv == 0:
            # Calc loss
            losses_valid, logits_concat_valid = get_loss_lstm(
                model, 
                x_batch_valid, 
                y_batch_valid, 
                training=False, 
                order_sprt=order_sprt,
                duration=duration)

            # Calc confusion matrix of multiplets for every frame
            seqconfmx_mult_valid =  multiplet_sequential_confmx(
                logits_concat_valid, y_batch_valid)

            # Calc confusion matrix of log-likelihood ratios for every frame
            seqconfmx_llr_valid = binary_llr_sequential_confmx(
                logits_concat_valid, y_batch_valid)

            # Confusion matrix of SPRT and mean hitting time of a batch
            dict_confmx_sprt_valid, dict_mean_hittimes_valid,\
                dict_var_hittimes_valid,  dict_truncate_rates_valid =\
                run_truncated_sprt(
                    list_alpha, list_beta, logits_concat_valid, y_batch_valid)

        else:
            losses_valid_tmp, logits_concat_valid = get_loss_lstm(
                model, 
                x_batch_valid, 
                y_batch_valid, 
                training=False, 
                order_sprt=order_sprt,
                duration=duration)
            seqconfmx_mult_valid += multiplet_sequential_confmx(
                logits_concat_valid, y_batch_valid)
            seqconfmx_llr_valid += binary_llr_sequential_confmx(
                logits_concat_valid, y_batch_valid)

            dict_confmx_sprt_valid_tmp, dict_mean_hittimes_valid_tmp,\
                dict_var_hittimes_valid_tmp, dict_truncate_rates_valid_tmp =\
                run_truncated_sprt(
                    list_alpha, list_beta, logits_concat_valid, y_batch_valid)

            for iter_idx in range(len(losses_valid)):
                losses_valid[iter_idx] += losses_valid_tmp[iter_idx]

            for key in dict_confmx_sprt_valid_tmp.keys():
                dict_confmx_sprt_valid[key] += dict_confmx_sprt_valid_tmp[key]
                dict_mean_hittimes_valid[key] += dict_mean_hittimes_valid_tmp[key]
                dict_var_hittimes_valid[key] += dict_var_hittimes_valid_tmp[key]
                dict_truncate_rates_valid[key] += dict_truncate_rates_valid_tmp[key]

        # Verbose
        if ((iter_bv+1)%10 == 0) or (iter_bv == 0):
            sys.stdout.write(
                "\rValidation Iter: {:3d}/{:3d}".format(
                    iter_bv+1, (num_validdata//batch_size) + 1))
            sys.stdout.flush()

    # Normalization and calc metrics from confmx
    for iter_idx in range(len(losses_valid)):
        losses_valid[iter_idx] /= cnt

    list_metrics_mult_valid = seqconfmx_to_list_metrics(seqconfmx_mult_valid)
    list_metrics_llr_valid = seqconfmx_to_list_metrics(seqconfmx_llr_valid)
    dict_metrics_sprt_valid = dict_confmx_to_dict_metrics(dict_confmx_sprt_valid)
    list_bac_llr_valid = list_metrics_to_list_bac(list_metrics_llr_valid)
    mean_bac_valid = np.mean(list_bac_llr_valid)

    for key in dict_confmx_sprt_valid.keys():
        dict_mean_hittimes_valid[key] /= cnt
        dict_var_hittimes_valid[key] /= cnt        
        dict_truncate_rates_valid[key] /= cnt                

    return losses_valid, list_metrics_mult_valid, list_metrics_llr_valid,\
        dict_metrics_sprt_valid,dict_mean_hittimes_valid,\
        dict_var_hittimes_valid, dict_truncate_rates_valid,  mean_bac_valid


# Main Function
def objective(trial):
    # Timestamp
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]

    # Suggest parameters if necessary
    ####################################
    assert (config["exp_phase"] == "tuning") or (config["exp_phase"] == "stat")\
        or (config["exp_phase"] == "try")

    if config["exp_phase"] == "tuning":
        list_suggest = suggest_parameters_fe(
            trial, 
            list_lr=config["list_lr"], 
            list_bs=config["list_bs"], 
            list_opt=config["list_opt"], 
            list_do=config["list_do"], 
            list_wd=config["list_wd"])

        print("##############################################################")
        print("Suggest params: ", list_suggest)
        print("##############################################################")

        learning_rate = list_suggest[0]
        batch_size = list_suggest[1]
        name_optimizer = list_suggest[2]
        dropout = list_suggest[3]
        weight_decay = list_suggest[4]

        config["learning_rates"] = [learning_rate, learning_rate*0.1]
        config["batch_size"] = batch_size
        config["name_optimizer"] = name_optimizer
        config["dropout"] = dropout
        config["weight_decay"] = weight_decay

    # Load data
    ##################################
    # Reed tfr and make
    parsed_image_dataset_train, parsed_image_dataset_valid, parsed_image_dataset_test = \
    read_tfrecords_nosaic_mnist(
        record_file_train=config["tfr_train"], 
        record_file_test=config["tfr_test"], 
        batch_size=config["batch_size"], 
        shuffle_buffer_size=2000)
        
    # Model
    ######################################
    model = LSTMModelLite(
        config["nb_cls"], 
        config["width_lstm"], 
        dropout=config["dropout"], 
        activation=config["activation"])
 
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
                x_batch, y_batch = decode_feat(
                    feats, config["duration"], config["feat_dim"], 
                    dtype_feat=tf.float32, dtype_label=tf.int32) 

                # Show summary of model
                if (epoch == 0) and (iter_b == 0):
                    model.build(input_shape=x_batch.shape)
                    model.summary() 

                # Calc loss and grad, and backpropagation
                grads, losses, logits_concat = get_gradient_lstm(
                    model, 
                    x_batch, 
                    y_batch, 
                    training=True, 
                    order_sprt=config["order_sprt"],
                    duration=config["duration"],
                    param_multiplet_loss=config["param_multiplet_loss"], 
                    param_llr_loss=config["param_llr_loss"], 
                    param_wd=config["weight_decay"], 
                    flag_wd=flag_wd_in_loss)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                global_step.assign_add(1)
                
                #Verbose
                if tf.equal(global_step % config["train_display_step"], 0)\
                    or tf.equal(global_step, 1):
                    print("Global Step={:7d} Epoch={:4d}/{:4d} Iter={:5d}/{:5d}: sum loss={:7.5f}, multiplet loss={:7.5f}, llr loss={:7.5f}".format(
                        int(global_step), 
                        epoch+1, 
                        config["nb_epochs"], 
                        iter_b+1, 
                        (config["num_traindata"]//config["batch_size"])+1, 
                        losses[1]+losses[2], 
                        losses[1], 
                        losses[2]))

                    # Confusion matrix of multiplets for every frame
                    seqconfmx_mult = multiplet_sequential_confmx(logits_concat, y_batch)
                    list_metrics_mult = seqconfmx_to_list_metrics(seqconfmx_mult)

                    # Confusion matrix of log-likelihood ratios for every frame
                    seqconfmx_llr = binary_llr_sequential_confmx(logits_concat, y_batch)
                    list_metrics_llr = seqconfmx_to_list_metrics(seqconfmx_llr)


                    # Confusion matrix of SPRT and mean hitting time of a batch
                    dict_confmx_sprt, dict_mean_hittimes, _, dict_truncate_rates = \
                        run_truncated_sprt(
                            config["list_alpha"], 
                            config["list_beta"], 
                            logits_concat, 
                            y_batch, 
                            verbose=True)
                    dict_metrics_sprt = dict_confmx_to_dict_metrics(dict_confmx_sprt)

                    # Tensorboard
                    tblog_writer_train(
                        tblogger, 
                        losses, 
                        global_step, 
                        dict_metrics_sprt, 
                        dict_mean_hittimes, 
                        dict_truncate_rates, 
                        list_metrics_mult, 
                        list_metrics_llr)

                # Validation
                if tf.equal(global_step % config["valid_step"], 0) or\
                    tf.equal(global_step, 1):
                    losses_valid, list_metrics_mult_valid, list_metrics_llr_valid,\
                        dict_metrics_sprt_valid,dict_mean_hittimes_valid, _,\
                        dict_truncate_rates_valid,  mean_bac_valid = validation_loop(
                            parsed_image_dataset_valid, 
                            model, 
                            config["order_sprt"],
                            config["duration"], 
                            config["list_alpha"], 
                            config["list_beta"], 
                            config["num_validdata"], 
                            config["batch_size"],
                            config["feat_dim"])

                    # Tensorboard for validation
                    tblog_writer_valid(
                        tblogger, 
                        losses_valid, 
                        global_step, 
                        mean_bac_valid, 
                        dict_metrics_sprt_valid, 
                        dict_mean_hittimes_valid, 
                        dict_truncate_rates_valid, 
                        list_metrics_mult_valid, 
                        list_metrics_llr_valid)           

                    # For exp_phase="tuning", optuna
                    if best < mean_bac_valid:
                        best = mean_bac_valid
                    
                        # Save checkpoint
                        ckpt_manager._checkpoint_prefix = \
                            ckpt_manager._checkpoint_prefix[:ckpt_manager._checkpoint_prefix.rfind("/") + 1] +\
                            "ckpt_step{}_mbac{:.5f}".format(int(global_step), best)
                        save_path_prefix = ckpt_manager.save()
                        print("Saved checkpoint for step {}: {}".format(
                            int(global_step), save_path_prefix))

                        # Test
                        ############################
                        if config["exp_phase"] == "stat":
                            # Test loop
                            _, _, _,dict_metrics_sprt_test,dict_mean_hittimes_test,\
                                dict_var_hittimes_test, dict_truncate_rates_test,\
                                mean_bac_test = validation_loop(
                                    parsed_image_dataset_test, 
                                    model, 
                                    config["order_sprt"], 
                                    config["duration"],
                                    config["list_alpha"], 
                                    config["list_beta"], 
                                    config["num_testdata"], 
                                    config["batch_size"],
                                    config["feat_dim"])
     
    # Final processes
    ###############################################
    if config["exp_phase"] == "stat":
        # Save best values to .db file 
        trial.set_user_attr("mean_bac", float(mean_bac_test))
        for key in dict_metrics_sprt_test.keys():
            metrics_test = dict_metrics_sprt_test[key]
            mean_hittime_test = float(dict_mean_hittimes_test[key])
            var_hittime_test = float(dict_var_hittimes_test[key])            
            truncate_rate_test = float(dict_truncate_rates_test[key])
            balanced_accuracy_test = float(metrics_test["BAC"][0])
            sensitivity_test = float(metrics_test["SNS"][0])
            specificity_test = float(metrics_test["SNS"][1])
            trial.set_user_attr("mean_hittime_" + key, mean_hittime_test)
            trial.set_user_attr("variance_hittime_" + key, var_hittime_test)
            trial.set_user_attr("truncate_rate_" + key, truncate_rate_test)
            trial.set_user_attr("balanced_accuracy_" + key, balanced_accuracy_test)
            trial.set_user_attr("TPR_" + key, sensitivity_test)
            trial.set_user_attr("TNR_" + key, specificity_test)

    # Return best valid result for optuna
    return 1 - best


if __name__ == '__main__':
    run_optuna(
        config["root_dblogs"], 
        config["subproject_name"], 
        config["exp_phase"], 
        objective, 
        config["nb_trials"])

