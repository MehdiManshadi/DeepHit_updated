import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random

from utils_eval import c_index

SEED = 13


def setSeed(seed=SEED):
    """Ensure reproducibility across runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


batch_loss_history = []
iteration_history = []


def TrainDeepHit(
    model,
    tr_data, tr_mask1, tr_mask2, tr_time, tr_label,
    eval_time,
    test_size=0.2,
    max_epochs=100,
    batch_size=128,
    learning_rate=1e-3,
    alpha=1.0,
    beta=1.0
):
    """
    Train DeepHit model with validation-based early stopping
    and time-dependent C-index evaluation.
    """

    # ==================================================
    # Optimizer
    # ==================================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_Category = tr_mask2.shape[1]

    # ==================================================
    # Train step
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
    # Train/Validation split
    # ==================================================
    (tr_data, te_data,
     tr_time, te_time,
     tr_label, te_label,
     tr_mask1, te_mask1,
     tr_mask2, te_mask2) = train_test_split(
        tr_data, tr_time, tr_label, tr_mask1, tr_mask2,
        test_size=test_size,
        random_state=SEED
    )

    num_Event = tr_mask1.shape[1]
    N = tr_data.shape[0]

    print("\n===== START TRAINING =====\n")

    global_iter = 0
    best_test_loss = np.inf
    patience = 3
    best_cindex = -np.inf

    # ==================================================
    # Training loop
    # ==================================================
    for epoch in range(max_epochs):
        # reset metrics for this epoch
        epoch_losses = []
        epoch_ll = []
        epoch_rank = []
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

            epoch_losses.append(loss.numpy())
            epoch_ll.append(ll.numpy())
            epoch_rank.append(rank.numpy())

            batch_loss_history.append(loss.numpy())
            iteration_history.append(global_iter)
            global_iter += 1
        
        # ==================================================
        # Validation loss (early stopping)
        # ==================================================
        teLoss, _, _ = testStep(
            te_data,
            te_mask1,
            te_mask2,
            te_label,
            te_time
        )

        teLoss= teLoss.numpy()

        if teLoss < best_test_loss:
            best_test_loss = teLoss
            best_weights = model.get_weights()
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # ==================================================
        # C-index evaluation
        # ==================================================
        pred = model.call(te_data, training=False).numpy()
        resultfeat = np.zeros([num_Event, len(eval_time)])

        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time)

            if eval_horizon >= num_Category:
                print("ERROR: evaluation horizon is out of range")
                resultfeat[:, t] = -1
                continue

            risk = np.sum(pred[:, :, :(eval_horizon + 1)], axis=2)

            for k in range(num_Event):
                resultfeat[k, t] = c_index(
                    risk[:, k],
                    te_time,
                    (te_label[:, 0] == k + 1).astype(int),
                    eval_horizon
                )

        print(
            f"Epoch {epoch:03d} | "
            f"Train {np.mean(epoch_losses):.4f} "
            f"(LL {np.mean(epoch_ll):.4f}, Rank {np.mean(epoch_rank):.4f}) | "
            f"Test {teLoss:.4f} | "
            f"Test C-index {resultfeat[0, 0]:.4f} "
            f"(competing event: {resultfeat[1, 0]:.4f})"
        )

    model.set_weights(best_weights) # Restore best model weights
    return model