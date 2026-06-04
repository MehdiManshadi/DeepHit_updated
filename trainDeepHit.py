import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils_eval import c_index
from sklearn.model_selection import train_test_split
import os
import random
SEED = 13


def setSeed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"   # deterministic TF ops
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

batch_loss_history = []
iteration_history = []


def TrainDeepHit(model, tr_data, tr_mask1, tr_mask2, tr_time, tr_label,
                  va_data, va_mask1, va_mask2, va_time, va_label, eval_time, test_size = 0.2, max_epochs = 100, batch_size = 128, learning_rate = 1e-3, alpha=1.0, beta=1.0):
    # ==================================================
    # Optimizer
    # ==================================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_Category = tr_mask2.shape[1] 

    # ==================================================
    # Training step
    # ==================================================
    @tf.function
    def trainStep(x, mask1, mask2, label, timeIndex):
        with tf.GradientTape() as tape:
            pmf = model(x, training=True)

            totalLoss, lossLL, lossRank = model.total_loss(
                pmf,
                mask1,
                mask2,
                label,
                timeIndex,
                alpha=alpha,
                beta=beta
            )

        grads = tape.gradient(totalLoss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return totalLoss, lossLL, lossRank

    # ==================================================
    # Validation step
    # ==================================================
    @tf.function
    def testStep(x, mask1, mask2, label, timeIndex):
        pmf = model(x, training=False)

        totalLoss, lossLL, lossRank = model.total_loss(
            pmf,
            mask1,
            mask2,
            label,
            timeIndex,
            alpha=alpha,
            beta=beta
        )

        return totalLoss, lossLL, lossRank

    # ==================================================
    # Training loop
    # ==================================================
    (tr_data, te_data,
    tr_time, te_time,
    tr_label, te_label,
    tr_mask1, te_mask1,
    tr_mask2, te_mask2
    ) = train_test_split(
    tr_data, tr_time, tr_label, tr_mask1, tr_mask2,
    test_size=test_size,
    random_state=SEED)


    num_Event = tr_mask1.shape[1]

    N = tr_data.shape[0]

    print("\n===== START TRAINING =====\n")

    train_losses = []
    train_ll = []
    train_rank = []
    global_iter = 0
    best_train_loss = 100
    best_cindex = 0
    best_model = None
    break_limit = 3
    for epoch in range(max_epochs):
        # shuffle indices (reproducible)
        idx = np.random.permutation(N)

        for i in range(0, N, batch_size):
            b = idx[i:i + batch_size]

            loss, ll, rank = trainStep(
                tr_data[b],
                tr_mask1[b],
                tr_mask2[b],
                tr_label[b],
                tr_time[b]
            )

            train_losses.append(loss.numpy())
            train_ll.append(ll.numpy())
            train_rank.append(rank.numpy())

            batch_loss_history.append(loss.numpy())
            iteration_history.append(global_iter)
            global_iter += 1

        # validation (full batch)
        teLoss, teLL, teRank = testStep(
            te_data,
            te_mask1,
            te_mask2,
            te_label,
            te_time
        )
        if(teLoss < best_train_loss):
            best_train_loss = teLoss
            break_limit = 3
        else:
            break_limit = break_limit - 1

        if(break_limit == 0):
            break

        pred = model.call(va_data)
        resultfeat = np.zeros([num_Event, len(eval_time)])

        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time)

            if eval_horizon >= num_Category:
                print('ERROR: evaluation horizon is out of range')
                resultfeat[:, t] = -1
            else:
                # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
                risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until eval_time

                # Calculate permutation importance for each eval horizon and event
                for k in range(num_Event):
                    resultfeat[k, t] = c_index(risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                    
                    #result.loc[k+1, feat, perm_iter, "Delta t = " + "{:03d}".format(t_time)] = result1[k,t] - resultfeat[k,t]
                    # since we compare risk_scores, the true label is that event occurs before time horizon
                    #result.loc[k+1, feat, perm_iter, "Delta t = " + "{:03d}".format(t_time)] = result1[k,t] - resultfeat[k,t]
                    # since we compare risk_scores, the true label is that event occurs before time horizon
        
        if(resultfeat[0,0] > best_cindex):
            best_cindex = resultfeat[0,0]
            best_cindex_comp = resultfeat[1,0]
            best_model = model

        print(
            f"Epoch {epoch:03d} | "
            f"Train {np.mean(train_losses):.4f} "
            f"(LL {np.mean(train_ll):.4f}, Rank {np.mean(train_rank):.4f}) | "
            f"Val {teLoss.numpy():.4f}"
        )
    
    print(f"\nBest C-index at eval time: {best_cindex:.3f} (competing event: {best_cindex_comp:.3f})\n")
    return best_model