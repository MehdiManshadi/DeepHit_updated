import tensorflow as tf

class DeepHitPlus(tf.keras.Model):
    def __init__(
        self,
        num_event,
        num_bins,
        shared_layers=(64, 64),
        MR_layers=(64, 64),
        CR_layers=(64, 64),
        dropout=0.5
    ):
        super().__init__()

        self.num_event = num_event
        self.num_bins = num_bins

        # shared network
        self.shared_net = tf.keras.Sequential([
            tf.keras.layers.Dense(h, activation="relu")
            for h in shared_layers
        ])

        # event-specific networks
        self.event_nets = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(h, activation="relu")
                for h in MR_layers
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Dense(h, activation="relu")
                for h in CR_layers
            ])
            
        ]

        self.dropout = tf.keras.layers.Dropout(dropout)

        # output: event × time bins
        self.out_layer = tf.keras.layers.Dense(
            num_event * num_bins, activation="softmax"
        )

    def call(self, x, training=False):
        shared = self.shared_net(x, training=training)
        shared = tf.concat([shared, x], axis=1)
        
        event_outs = []
        for k in range(self.num_event):
            ek = self.event_nets[k](shared, training=training)
            event_outs.append(ek)

        h = tf.concat(event_outs, axis=1)
        h = self.dropout(h, training=training)

        logits = self.out_layer(h)
        pmf = tf.reshape(
            logits, (-1, self.num_event, self.num_bins)
        )

        return pmf

    # --------------------------------------------------
    # LOSS 1 — Log-likelihood (faithful DeepHitPlus)
    # --------------------------------------------------
    def loss_log_likelihood(self, pmf, mask1, label, eps=1e-8):
        """
        pmf   : [B, K, T]
        mask1 : [B, K, T]
        label : [B, 1]  (0=censored, 1..K=event)
        """
        I1 = tf.cast(label > 0, tf.float32)   # event indicator

        tmp = tf.reduce_sum(
            tf.reduce_sum(mask1 * pmf, axis=2),
            axis=1,
            keepdims=True
        )

        log_tmp = tf.math.log(tmp + eps)

        loss = -tf.reduce_mean(I1 * log_tmp + (1.0 - I1) * log_tmp)
        return loss

    # --------------------------------------------------
    # LOSS 2 — Ranking loss (faithful DeepHitPlus)
    # --------------------------------------------------
    def loss_ranking(self, pmf, mask2, label, timeIndex, sigma=0.1):
        """
        pmf       : [B, K, T]
        mask2     : [B, T]
        label     : [B, 1]
        timeIndex : [B, 1]
        """
        B = tf.shape(pmf)[0]
        one = tf.ones((B, 1), dtype=tf.float32)

        eta_all = []

        for e in range(self.num_event):
            # indicator for event e+1
            I2 = tf.cast(label == (e + 1), tf.float32)
            I2 = tf.linalg.diag(tf.squeeze(I2))

            # event-specific joint PMF
            tmp_e = pmf[:, e, :]                        # [B, T]

            # risk matrix
            R = tf.matmul(tmp_e, mask2, transpose_b=True)
            diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1])

            R = tf.transpose(diag_R) - R
            R = tf.transpose(R)

            # time ordering matrix
            Tmat = tf.cast(timeIndex < tf.transpose(timeIndex), tf.float32)
            Tmat = tf.matmul(I2, Tmat)

            eta = tf.reduce_mean(
                Tmat * tf.exp(-R / sigma),
                axis=1,
                keepdims=True
            )
            eta_all.append(eta)

        eta_all = tf.stack(eta_all, axis=1)      # [B, K, 1]
        eta_all = tf.reduce_mean(
            tf.reshape(eta_all, [-1, self.num_event]),
            axis=1,
            keepdims=True
        )

        return tf.reduce_sum(eta_all)
    

    # --------------------------------------------------
    # TOTAL LOSS — DeepHitPlus
    # --------------------------------------------------
    def total_loss(
        self,
        pmf,
        mask1,
        mask2,
        label,
        timeIndex,
        alpha=1.0,
        beta=1.0
    ):
        """
        alpha: weight for log-likelihood loss
        beta : weight for ranking loss
        """

        lossLL = self.loss_log_likelihood(pmf, mask1, label)
        lossRank = self.loss_ranking(pmf, mask2, label, timeIndex)

        total = alpha * lossLL + beta * lossRank

        return total, lossLL, lossRank
