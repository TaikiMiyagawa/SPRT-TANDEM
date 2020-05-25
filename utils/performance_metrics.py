import tensorflow as tf
import numpy as np

def confmx_to_metrics(confmx):
    """Calc confusion-matrix-based performance metrics.
    Args:
        confmx: A confusion matrix Tensor with shape (num classes, num classes).
    Return:
        dict_metrics: A dictionary of dictionaries of performance metrics. E.g., sensitivity of class 3 is dics_metrics["SNS"][3], and accuracy is dict_metrics["ACC"]["original"]
    Remark:
        - SNS: sensitivity, recall, true positive rate
        - SPC: specificity, true negative rate
        - PRC: precision
        - ACC: accuracy
        - BAC: balanced accuracy
        - F1: F1 score
        - GM: geometric mean
        - MCC: Matthews correlation coefficient
        - MK: markedness
        - e.g., The classwise accuracy of class i is dict_metric["SNS"][i].
        - "dict_metrics" contains some redundant metrics; e.g., for binary classification, dict_metric["SNS"]["macro"] = dict_metric["BAC"][0] = dict_metric["BAC"][1] = ...
        - Macro-averaged metrics are more robust to class-imbalance than micro-averaged ones, but note that macro-averaged metrics are sometimes equal to be ACC.
        - Most of the micro-averaged metrics are equal to or equivalent to ACC.
    """
    confmx = tf.cast(confmx, tf.int64) # prevent from overflow
    nb_cls = confmx.shape[0]
    dict_metrics = {
        "SNS":dict(),
        "SPC":dict(),
        "PRC":dict(),
        "ACC":dict(),
        "BAC":dict(),
        "F1":dict(),
        "GM":dict(),
        "MCC":dict(),
        "MK":dict()
    }
    TP_tot = 0
    TN_tot = 0
    FP_tot = 0
    FN_tot = 0

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix
    for i in range(nb_cls):
        # Initialization
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # Calc TP, TN, FP, FN for class i
        TP = confmx[i,i]
        for j in range(nb_cls):
            if j == i:
                continue
            FP += confmx[j,i]
            FN += confmx[i,j]
            for k in range(nb_cls):
                if k == i:
                    continue
                TN += confmx[j,k]

        # Calc performance metrics of class i
        dict_metrics["SNS"][i] = TP/(TP+FN) if TP+FN != 0 else 0.
        dict_metrics["SPC"][i] = TN/(TN+FP) if TN+FP != 0 else 0.
        dict_metrics["PRC"][i] = TP/(TP+FP) if TP+FP != 0 else 0.
        dict_metrics["ACC"][i] = (TP+TN)/(TP+FN+TN+FP) if TP+FN+TN+FP != 0 else 0.
        dict_metrics["BAC"][i] = (dict_metrics["SNS"][i] + dict_metrics["SPC"][i])/2
        dict_metrics["F1"][i] = 2*(dict_metrics["PRC"][i] * dict_metrics["SNS"][i]) / (dict_metrics["PRC"][i] + dict_metrics["SNS"][i]) if dict_metrics["PRC"][i] + dict_metrics["SNS"][i] != 0 else 0.
        dict_metrics["GM"][i] = np.sqrt(dict_metrics["SNS"][i] * dict_metrics["SPC"][i])
        dict_metrics["MCC"][i] = ((TP*TN) - (FP*FN))/(np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )) if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0 else 0.
        dict_metrics["MK"][i] = dict_metrics["PRC"][i] + (TN/(TN+FN)) - 1 if TN+FN != 0 else dict_metrics["PRC"][i] - 1

        TP_tot += TP
        TN_tot += TN
        FP_tot += FP
        FN_tot += FN

    # Calc micro- and macro- averaged metrics
    # sometimes returns nan. please fix it
    dict_metrics["SNS"]["macro"] = np.mean([dict_metrics["SNS"][i] for i in range(nb_cls)])
    dict_metrics["SNS"]["micro"] = TP_tot/(TP_tot+FN_tot) if TP_tot+FN_tot != 0 else 0. # = original ACC. 
    dict_metrics["SPC"]["macro"] = np.mean([dict_metrics["SPC"][i] for i in range(nb_cls)])
    dict_metrics["SPC"]["micro"] = TN_tot/(TN_tot+FP_tot) if TN_tot+FP_tot != 0 else 0.
    dict_metrics["PRC"]["macro"] = np.mean([dict_metrics["PRC"][i] for i in range(nb_cls)])
    dict_metrics["PRC"]["micro"] = TP_tot/(TP_tot+FP_tot) if TP_tot+FP_tot != 0 else 0. # = original ACC. 
    dict_metrics["ACC"]["macro"] = np.mean([dict_metrics["ACC"][i] for i in range(nb_cls)])
    dict_metrics["ACC"]["micro"] = (TP_tot+TN_tot)/(TP_tot+FN_tot+TN_tot+FP_tot) if TP_tot+FN_tot+TN_tot+FP_tot != 0 else 0.
    dict_metrics["ACC"]["original"] = ((nb_cls/2) * dict_metrics["ACC"]["micro"]) - ((nb_cls-2)/2)
    dict_metrics["BAC"]["macro"] = np.mean([dict_metrics["BAC"][i] for i in range(nb_cls)])
    dict_metrics["BAC"]["micro"] = (dict_metrics["SNS"]["micro"] + dict_metrics["SPC"]["micro"])/2
    dict_metrics["F1"]["macro"] = np.mean([dict_metrics["F1"][i] for i in range(nb_cls)])
    dict_metrics["F1"]["micro"] = 2*(dict_metrics["PRC"]["micro"] * dict_metrics["SNS"]["micro"]) / (dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"]) if dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"] != 0 else 0.# = original ACC. 
    dict_metrics["GM"]["macro"] = np.mean([dict_metrics["GM"][i] for i in range(nb_cls)])
    dict_metrics["GM"]["micro"] = np.sqrt(dict_metrics["SNS"]["micro"] * dict_metrics["SPC"]["micro"])
    dict_metrics["MCC"]["macro"] = np.mean([dict_metrics["MCC"][i] for i in range(nb_cls)])
    dict_metrics["MCC"]["micro"] = ((TP_tot*TN_tot) - (FP_tot*FN_tot))/(np.sqrt( (TP_tot+FP_tot)*(TP_tot+FN_tot)*(TN_tot+FP_tot)*(TN_tot+FN_tot) )) if (TP_tot+FP_tot)*(TP_tot+FN_tot)*(TN_tot+FP_tot)*(TN_tot+FN_tot) != 0 else 0.
    dict_metrics["MK"]["macro"] = np.mean([dict_metrics["MK"][i] for i in range(nb_cls)])
    dict_metrics["MK"]["micro"] = dict_metrics["PRC"]["micro"] + (TN_tot/(TN_tot+FN_tot)) - 1 if TN_tot+FN_tot != 0 else 0. 

    return dict_metrics


def binary_confmx_to_bac(confmx):
    epsilon = 1e-10
    confmx = tf.cast(confmx, tf.float64)
    TP = confmx[0,0]
    TN = confmx[1,1]
    FN = confmx[0,1]
    FP = confmx[1,0]
    tpr = TP / (TP + FN + epsilon)
    tnr = TN / (TN + FP + epsilon)    
    bac = tf.cast(0.5 * (tpr + tnr), tf.float32)
    
    return bac


def logits_to_confmx(logits, labels):
    """Calculate the confusion matrix from logits.
    Args: 
        logits: A logit Tensor with shape (batch, num classes).
        labels: A non-one-hot label Tensor with shape (batch,).
    Returns:
        confmx: A Tensor with shape (num classes, num classes).
    """
    logits_shape = logits.shape # (batch, num classes)
    nb_cls = logits_shape[-1]

    # First order_sprt+1 frames
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32) # (batch,)
    confmx = tf.math.confusion_matrix(
        labels=labels, predictions=preds, num_classes=nb_cls, dtype=tf.int32)

    return confmx


def dict_confmx_to_dict_metrics(dict_confmx):
    """Transforms a dictionary of confusion matrices to a dictionary (with the same key as dict_confmx) of dictionaries of metrics."""
    dict_metrics = dict()
    for key, confmx in dict_confmx.items():
        dict_metrics[key] = confmx_to_metrics(confmx)

    return dict_metrics


def seqconfmx_to_list_metrics(seqconfmx):
    """Transforms a Tensor of confusion matirces with shape (LEN, num classes, num classes) to a list (with length LEN) of dictionaries of metrics, where LEN is undetermined."""
    sequence_length = seqconfmx.shape[0]
    list_metrics = [None]*sequence_length
    for iter_idx in range(sequence_length):
        list_metrics[iter_idx] = confmx_to_metrics(seqconfmx[iter_idx])        

    return list_metrics


def list_metrics_to_list_bac(list_metrics):
    """
    Arg:
        list_metrics: A list of dictionaries. 
    Return:
        list_bacs: A list of floats with the same length as list_metric's.
    """
    list_bacs = [None]*len(list_metrics)
    for iter_idx, iter_dict in enumerate(list_metrics):
        bac = iter_dict["BAC"][0]
        list_bacs[iter_idx] = bac

    return list_bacs


def multiplet_sequential_confmx(logits_concat, labels_concat):
    """Calculate the confusion matrix for each frame from logits. Lite.
    Args: 
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, nb_cls). This is the output of datasets.data_processing.sequential_concat(logit_slice, y_slice).
        labels_concat: A non-one-hot label Tensor with shape (batch,). This is the output of the function datasets.data_processing.sequential_conclogit_slice, y_slice).
    Return:
        seqconfmx_mult: A Tensor with shape (duration, num classes, num classes). This is the series of confusion matrices computed from multiplets.
    Remark:
        e.g., order_sprt = 5, duration = 20:
            confusion matrix for frame001 is given by the 001let of frame001
            confusion matrix for frame002 is given by the 002let of frame002
            ...
            confusion matrix for frame005 is given by the 004let of frame004
            confusion matrix for frame005 is given by the 005let of frame005
            confusion matrix for frame006 is given by the 005let of frame006 computed from frame002-006
            confusion matrix for frame007 is given by the 005let of frame007 computed from frame003-007
            ...
            confusion matrix for frame019 is given by the 005let of frame019 computed from frame015-019
            confusion matrix for frame020 is given by the 005let of frame020 computed from frame016-020
    """
    logits_concat_shape = logits_concat.shape # (batch, (duration - order_sprt), order_sprt + 1, num classes)
    nb_cls = logits_concat_shape[-1]

    # First order_sprt+1 frames
    logits_concat_former = logits_concat[:,0,:,:] # (batch, order_sprt + 1, num classes)

    for iter_idx in range(logits_concat_shape[2]):
        preds_former = tf.argmax(logits_concat_former[:,iter_idx,:], axis=-1, output_type=tf.int32) # (batch,)
        if iter_idx == 0:
            seqconfmx_mult = tf.math.confusion_matrix(labels=labels_concat, predictions=preds_former, num_classes=nb_cls,
            dtype=tf.int32)
            seqconfmx_mult = tf.expand_dims(seqconfmx_mult, 0)
        else:
            seqconfmx_mult = tf.concat(
                [seqconfmx_mult, tf.expand_dims(tf.math.confusion_matrix(labels=labels_concat, predictions=preds_former, num_classes=nb_cls, dtype=tf.int32), 0)],
                axis=0
                )

    # Latter duration-order_sprt-1 frames
    logits_concat_latter = logits_concat[:,1:,-1,:] # (batch, duration-order_sprt-1, num classes)

    for iter_idx in range(logits_concat_shape[1]-1):
        preds_latter = tf.argmax(logits_concat_latter[:,iter_idx,:], axis=-1, output_type=tf.int32) # (batch,)
        seqconfmx_mult = tf.concat(
            [seqconfmx_mult, tf.expand_dims(tf.math.confusion_matrix(labels=labels_concat, predictions=preds_latter, num_classes=nb_cls, dtype=tf.int32), 0)],
            axis=0
            )

    return seqconfmx_mult


def calc_binary_llrs(logits_concat):
    """Calculate the frame-by-frame log-likelihood ratios.
    Args:
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, 2). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Returns:
        A Tensor with shape (batch, duration). All the LLRs.
    Remarks:
        - The LLRs returned are the LLRs used in the "order_sprt"-th order SPRT; the LLRs unnecessary to calculate the "order_sprt"-th order SPRT are not included. 
        - "duration" and "order_sprt" are automatically calculated using logits_concat.shape.
        - Binary classification (num classes = 2) is assumed.
    """
    logits_concat_shape = logits_concat.shape

    # Start calc of LLRs. See the N-th-order SPRT formula.
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)
    list_llrs = []

    # i.i.d. SPRT (0th-order SPRT) (order_sprt=0)
    if order_sprt == 0:
        llrs_all_frames = logits_concat[:, :, order_sprt, 1] - logits_concat[:, :, order_sprt, 0] # (batch, duration-order_sprt, order_sprt+1, nb_cls=2) -> (batch, duration-order_sprt)
        for iter_frame in range(duration):
            llrs = tf.reduce_sum(llrs_all_frames[:, :iter_frame+1], -1) # (batch,)
            list_llrs.append(tf.expand_dims(llrs, 1))

    # N-th-order SPRT (order_sprt=N > 0)
    else:
        for iter_frame in range(duration):
            if iter_frame < order_sprt + 1:
                llrs = logits_concat[:, 0, iter_frame, 1] - logits_concat[:, 0, iter_frame, 0] 
                list_llrs.append(tf.expand_dims(llrs, 1))

            else:
                llrs1 = logits_concat[:, :iter_frame - order_sprt + 1, order_sprt, 1] - logits_concat[:, :iter_frame - order_sprt + 1, order_sprt, 0] # (batch, iter_frame-order_sprt)
                llrs1 = tf.reduce_sum(llrs1, -1) # (batch,)
                llrs2 = logits_concat[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 1] - logits_concat[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 0] # (batch, iter_frame-order_sprt-1)
                llrs2 = tf.reduce_sum(llrs2, -1) # (batch,)
                llrs = llrs1 - llrs2 # (batch, )
                list_llrs.append(tf.expand_dims(llrs, 1))

    return tf.concat(list_llrs, 1) # (batch, duration)


def binary_llr_sequential_confmx(logits_concat, labels_concat):
    """Calculate the frame-by-frame confusion matrices based on the log-likelihood ratios.
    Args:
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, 2). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        labels_concat: A non-one-hot label Tensor with shape (batch,). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Returns:
        seqconfmx_llr: A Tensor with shape (duration, 2, 2). The confusion matrices of forced decisions for all frames.
    Remark:
        Binary classification (num classes = 2) is assumed.
    """
    assert tf.reduce_max(labels_concat) <= 1, "Only nb_cls=2 is allowed."
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)

    # Calc log-likelihood ratios
    llrs = calc_binary_llrs(logits_concat) # (batch, duration)

    # Calc confusion matrices
    preds = tf.cast(tf.round(tf.nn.sigmoid(llrs)), tf.int32) # (batch, duration)        
    for iter_frame in range(duration):
        if iter_frame == 0:
            seqconfmx_llr = tf.math.confusion_matrix(labels=labels_concat, predictions=preds[:,iter_frame], num_classes=2, dtype=tf.int32)
            seqconfmx_llr = tf.expand_dims(seqconfmx_llr, 0)
        else:
            seqconfmx_llr = tf.concat(
                [seqconfmx_llr, 
                tf.expand_dims(
                    tf.math.confusion_matrix(
                        labels=labels_concat, 
                        predictions=preds[:,iter_frame],
                        num_classes=2, 
                        dtype=tf.int32), 
                    axis=0)],
                axis=0)

    return seqconfmx_llr
    

def binary_truncated_sprt(logits_concat, labels_concat, alpha, beta):
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)
    batch_size = labels_concat.shape[0]
    assert batch_size != 0
    
    # Calc thresholds
    thresh = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)]
    if not ( (thresh[1] >= thresh[0]) and (thresh[1] * thresh[0] < 0) ):
        raise ValueError(
            "thresh must be thresh[1] >= thresh[0] and thresh[1] * thresh[0] < 0. Now thresh = {}".format(thresh))

    # Calc log-likelihood ratios
    llrs = calc_binary_llrs(logits_concat) # (batch, duration)

    # Calc all predictions and waits
    signs1 = (tf.sign(llrs - thresh[1]) + 1)/2 #  1:hit, 0:wait
    signs0 = (-1 - tf.sign(thresh[0] - llrs))/2 # -1:hit, 0:wait
    preds_all_frames = signs1 + signs0 # (batch, duration), value= +1, 0, -1

    # Calc truncate rate
    hit_or_wait_all_frames = -(tf.abs(preds_all_frames) - 1) # wait=1, hit=0
    truncate_rate = tf.reduce_mean(tf.reduce_prod(hit_or_wait_all_frames, 1), 0)

    # Truncate survivors (forced decision)
    preds_last_frame = tf.sign(llrs[:,-1]) # (batch,) value= +1, -1
    preds_last_frame = tf.expand_dims(preds_last_frame, -1) # (batch, 1)
    preds_all_frames_trunc = tf.concat([preds_all_frames[:,:-1], preds_last_frame], -1) # (batch, duration-1)+(batch,1)=(batch, duration)

    if duration == 1:
        # Calc mean hitting time and confusion matrix
        mean_hittime = tf.constant(1., tf.float32)
        preds = preds_all_frames_trunc[:,0] # (batch,)
        preds = tf.cast((preds + 1) / 2, tf.int32)
        confmx = tf.math.confusion_matrix(labels_concat, preds, num_classes=2, dtype=tf.int32)

    else:
        # Calc mean hitting time
        mask = tf.constant([i+1 for i in range(duration)][::-1], tf.float32)
        mask = tf.tile(mask, [batch_size,])
        mask = tf.reshape(mask, [batch_size, duration])
        masked = preds_all_frames_trunc * mask # (batch, duration)

        signed_hittimes1 = tf.reduce_max(masked, 1, keepdims=True)
        signed_hittimes0 = tf.reduce_min(masked, 1, keepdims=True)
        signed_hittimes0_abs = tf.abs(signed_hittimes0)
        signed_hittimes_twin = tf.concat([signed_hittimes1, signed_hittimes0], 1)
        hittimes_twin = tf.abs(signed_hittimes_twin)

        answers1 = tf.greater(signed_hittimes1, signed_hittimes0_abs)
        answers0 = tf.less(signed_hittimes1, signed_hittimes0_abs)
        answers = tf.concat([answers1, answers0], 1)
        hittimes = hittimes_twin[answers]
        hittimes = duration - hittimes + 1
        mean_hittime, var_hittime = tf.nn.moments(hittimes, axes=[0])

        # Calc confusion matrix
        signs_twin = tf.sign(signed_hittimes_twin)
        preds = signs_twin[answers]
        preds = tf.cast((preds + 1) / 2, tf.int32)
        confmx = tf.math.confusion_matrix(labels_concat, preds, num_classes=2, dtype=tf.int32)

    return confmx, mean_hittime, var_hittime, truncate_rate


def run_truncated_sprt(list_alpha, list_beta, logits_concat, labels_concat, verbose=False):
    """ Calculate confusion matrix, mean hitting time, and truncate rate of a batch.
    Args:
        list_alpha: A list of floats.
        list_beta: A list of floats with the same length as list_alpha's.
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, 2). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A binary label Tensor with shape (batch size,) with label = 0 or 1. This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Returns:
        dict_confmx_sprt: A dictionary with keys like "thresh=0.2342,-0.2342". Value is a confusion matrix Tensor. 
        dict_mean_hittimes: A dictionary with keys like "thresh=0.2342,-0.2342". Value is a mean hitting time.
        dict_truncate_rates: A dictionary with keys like "thresh=0.2342,-0.2342". Value is an truncate rate.
    """
    dict_confmx_sprt = dict()
    dict_mean_hittimes = dict()
    dict_var_hittimes = dict()
    dict_truncate_rates = dict()
    batch_size_tmp = labels_concat.shape[0]

    for alpha, beta in zip(list_alpha, list_beta):
        # Calc thresholds
        alpha = float(alpha)
        beta = float(beta)
        thresh = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)]
        key = "thresh={:6.4f},{:7.4f}".format(thresh[0], thresh[1])

        # Run truncated sprt
        confmx, mean_hittime, var_hittime, truncate_rate = binary_truncated_sprt(logits_concat, labels_concat, alpha, beta)
        
        dict_confmx_sprt[key] = confmx
        dict_mean_hittimes[key] = mean_hittime
        dict_var_hittimes[key] = var_hittime
        dict_truncate_rates[key] = truncate_rate

        if verbose:
            print("====================================")
            print("SPRT w/ alpha={}, beta={}".format(alpha, beta))
            print("Thresholds = {}".format(thresh))
            print("Confusion Matrix")
            print(confmx)
            print("Mean Hitting Time: {} +- {}".format(mean_hittime, tf.sqrt(var_hittime)))
            print("truncate: {} / {} = {}".format(tf.round(truncate_rate*batch_size_tmp), batch_size_tmp, truncate_rate))
            print("====================================")

    return dict_confmx_sprt, dict_mean_hittimes, dict_var_hittimes, dict_truncate_rates


def binary_truncated_sprt_with_llrs(llrs, labels, alpha, beta, order_sprt):
    """ Used in run_truncated_sprt_with_llrs .
    Args:
        llrs: A Tensor with shape (batch, duration). LLRs (or scores) of all frames.
        labels: A Tensor with shape (batch,).
        alpha : A float.
        beta: A float.
        order_sprt: An int.
    Returns:
        confmx: A Tensor with shape (2, 2).
        mean_hittime: A scalar Tensor.
        var_hittime: A scalar Tensor.
        truncate_rate: A scalar Tensor.
    """
    llrs_shape = llrs.shape
    duration = int(llrs_shape[1])
    batch_size = llrs_shape[0]
    assert batch_size != 0
    
    # Calc thresholds
    thresh = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)]
    if not ( (thresh[1] >= thresh[0]) and (thresh[1] * thresh[0] < 0) ):
        raise ValueError("thresh must be thresh[1] >= thresh[0] and thresh[1] * thresh[0] < 0. Now thresh = {}".format(thresh))

    # Calc all predictions and waits
    signs1 = (tf.sign(llrs - thresh[1]) + 1)/2 #  1:hit, 0:wait
    signs0 = (-1 - tf.sign(thresh[0] - llrs))/2 # -1:hit, 0:wait
    preds_all_frames = signs1 + signs0 # (batch, duration), value= +1, 0, -1

    # Calc truncate rate
    hit_or_wait_all_frames = -(tf.abs(preds_all_frames) - 1) # wait=1, hit=0
    truncate_rate = tf.reduce_mean(tf.reduce_prod(hit_or_wait_all_frames, 1), 0)

    # Truncate survivors (forced decision)
    preds_last_frame = tf.sign(llrs[:,-1]) # (batch,) value= +1, -1
    preds_last_frame = tf.expand_dims(preds_last_frame, -1) # (batch, 1)
    preds_all_frames_trunc = tf.concat([preds_all_frames[:,:-1], preds_last_frame], -1) # (batch, duration-1)+(batch,1)=(batch, duration)

    if duration == 1:
        # Calc mean hitting time and confusion matrix
        mean_hittime = tf.constant(1., tf.float32)
        preds = preds_all_frames_trunc[:,0] # (batch,)
        preds = tf.cast((preds + 1) / 2, tf.int32)
        confmx = tf.math.confusion_matrix(labels, preds, num_classes=2, dtype=tf.int32)

    else:
        # Calc mean hitting time
        mask = tf.constant([i+1 for i in range(duration)][::-1], tf.float32)
        mask = tf.tile(mask, [batch_size,])
        mask = tf.reshape(mask, [batch_size, duration])
        masked = preds_all_frames_trunc * mask # (batch, duration)

        signed_hittimes1 = tf.reduce_max(masked, 1, keepdims=True)
        signed_hittimes0 = tf.reduce_min(masked, 1, keepdims=True)
        signed_hittimes0_abs = tf.abs(signed_hittimes0)
        signed_hittimes_twin = tf.concat([signed_hittimes1, signed_hittimes0], 1)
        hittimes_twin = tf.abs(signed_hittimes_twin)

        answers1 = tf.greater(signed_hittimes1, signed_hittimes0_abs)
        answers0 = tf.less(signed_hittimes1, signed_hittimes0_abs)
        answers = tf.concat([answers1, answers0], 1)
        hittimes = hittimes_twin[answers]
        hittimes = duration - hittimes + 1
        mean_hittime, var_hittime = tf.nn.moments(hittimes, axes=[0])

        # Calc confusion matrix
        signs_twin = tf.sign(signed_hittimes_twin)
        preds = signs_twin[answers]
        preds = tf.cast((preds + 1) / 2, tf.int32)
        confmx = tf.math.confusion_matrix(labels, preds, num_classes=2, dtype=tf.int32)

    return confmx, mean_hittime, var_hittime, truncate_rate


def run_truncated_sprt_with_llrs(list_alpha, list_beta, llrs, labels, order_sprt, verbose=False):
    """ Calculate confusion matrix, mean hitting time, and truncate rate of a batch.
    Args:
        list_alpha: A list of floats.
        list_beta: A list of floats with the same length as list_alpha's.
        llrs: A Tensor with shape (batch, duration). LLRs (or scores) of all frames.
        labels: A Tensor with shape (batch,). Labels with values 0 or 1. Binary classirfication is assumed.
        order_sprt: An int.
    Returns:
        dict_confmx_sprt: A dictionary with keys like "thresh=0.2342,-0.2342". Value is a confusion matrix Tensor. 
        dict_mean_hittimes: A dictionary with keys like "thresh=0.2342,-0.2342". Value is a mean hitting time.
        dict_truncate_rates: A dictionary with keys like "thresh=0.2342,-0.2342". Value is an truncate rate.
    """
    dict_confmx_sprt = dict()
    dict_mean_hittimes = dict()
    dict_var_hittimes = dict()
    dict_truncate_rates = dict()
    batch_size_tmp = labels.shape[0]

    for alpha, beta in zip(list_alpha, list_beta):
        # Calc thresholds
        alpha = float(alpha)
        beta = float(beta)
        thresh = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)]
        key = "thresh={:6.4f},{:7.4f}".format(thresh[0], thresh[1])

        # Run truncated sprt
        confmx, mean_hittime, var_hittime, truncate_rate = binary_truncated_sprt_with_llrs(
            llrs, 
            labels, 
            alpha, 
            beta,
            order_sprt)
        
        dict_confmx_sprt[key] = confmx
        dict_mean_hittimes[key] = mean_hittime
        dict_var_hittimes[key] = var_hittime
        dict_truncate_rates[key] = truncate_rate

        if verbose:
            print("====================================")
            print("SPRT w/ alpha={}, beta={}".format(alpha, beta))
            print("Thresholds = {}".format(thresh))
            print("Confusion Matrix")
            print(confmx)
            print("Mean Hitting Time: {} +- {}".format(mean_hittime, tf.sqrt(var_hittime)))
            print("truncate: {} / {} = {}".format(tf.round(truncate_rate*batch_size_tmp), batch_size_tmp, truncate_rate))
            print("====================================")

    return dict_confmx_sprt, dict_mean_hittimes, dict_var_hittimes, dict_truncate_rates


def binary_avescr_test(scores, labels, length=None, thresh=0.):
    """ Score Average Test. Not used.
    Args:
        scores: A Tensor with shape (batch, duration).
        labels: A Tensor with shape (batch,).
        length: A integer or None. If this is not None, it shouldb be 1 <= length <= duration, and scores after the length-th frame are thrown away.
        thresh: A float.
    Returns:
        confmx: A Tensor with shape (2, 2). Binary classification is assumed.
    """
    scores_shape = scores.shape
    duration = scores_shape[1]
    if not (length is None):
        assert 1 <= length <= duration
        scores = scores[:, :length]

    # Calc predictions
    score = tf.reduce_mean(scores, axis=1) - thresh
    preds = tf.round(tf.nn.sigmoid(score))

    # Calc confusion matrix
    confmx = tf.math.confusion_matrix(labels, preds, num_classes=2, dtype=tf.int32)

    return confmx


def binary_np_test(llrs, labels, length=None, thresh=0.):
    """ Neyman-Pearson Test. Make sure `llr` is the LLR.
    Args:
        llrs: A Tensor with shape (batch, duration).
        labels: A Tensor with shape (batch,).
        length: A integer or None. If this is not None, it shouldb be 1 <= length <= duration, and scores after the length-th frame are thrown away.
        thresh: A float.
    Returns:
        confmx: A Tensor with shape (2, 2). Binary classification is assumed.
    Remark:
        - Note that the NP test uses the likelihood ratio of a sequence, not a frame; that's why the input arg is LLRs, not scores, which is not equivalent to LLRs in general.
    """
    llrs_shape = llrs.shape
    duration = llrs_shape[1]
    if not (length is None):
        assert 1 <= length <= duration

    # Calc predictions
    llr = llrs[:,length-1] - thresh
    preds = tf.round(tf.nn.sigmoid(llr))

    # Calc confusion matrix
    confmx = tf.math.confusion_matrix(labels, preds, num_classes=2, dtype=tf.int32)

    return confmx