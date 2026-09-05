import numpy as np
# from .backprop import Tensor

def sinusoidal_positional_encoding(p_words:int, m_features:int):
    #n_sentences, p_words, m_features = embedding dims of each word are m_feature
    #P_E(pos,2i) = sin(pos/10000^(2i/n_dims))
    #P_E(pos,2i+1) = cos(pos/10000^(2i/n_dims))

    #division term = position/10000^(2i/n_dims) = position * 10000^-(2i/n_dims) = position * e^(-2*position*log(10000)/n_dims)
    position = np.arange(p_words)[:,np.newaxis]
    div_term = np.exp(-2*position*np.log(10000)/m_features)

    pos_embed = np.zeros(shape=(p_words,m_features))
    #even position 
    pos_embed[:,0::2] = np.sin(position*div_term)
    pos_embed[:,1::2] = np.cos(position*div_term)
    pos_embed = Tensor(pos_embed,label='pos_embed')
    return pos_embed

class attention_head:
    def __init__(self, shape: tuple[int, int] = (1, 1)):
        self.n, self.m_feature = shape
        std = np.sqrt(2 / self.m_feature)
        self.W_q = Tensor(np.random.randn(self.m_feature, self.m_feature) * std, label='w_q')
        self.W_k = Tensor(np.random.randn(self.m_feature, self.m_feature) * std, label='w_k')
        self.W_v = Tensor(np.random.randn(self.m_feature, self.m_feature) * std, label='w_v')

    def __call__(self, pos_aware_embed: Tensor):
        Q = pos_aware_embed * self.W_q
        K = pos_aware_embed * self.W_k
        V = pos_aware_embed * self.W_v

        div_factor = Tensor(1 / np.sqrt(self.n))
        similarity = div_factor.elementwise_mul(Q * K.swapaxes(-1, -2))

        attn_weights = similarity.softmax()          # normalize FIRST
        context_embed = attn_weights * V              # THEN weight values

        context_embed = context_embed + pos_aware_embed
        return context_embed

    def parameters(self):                              
        return self.W_q, self.W_k, self.W_v

class Transformer(MLP):
    def __init__(self, n_heads: int = 8, p_words: int = None, m_features: int = 96):
        self.n_heads = n_heads
        self.lr = 0.01
        self.m_features = m_features
        self.p_words = p_words

        std = np.sqrt(2 / m_features)
        self.all_attention_heads = [attention_head(shape=(p_words, m_features)) for _ in range(self.n_heads)]
        self.W_multi_head = Tensor(np.random.randn(m_features * self.n_heads, m_features) * std, label="multihead")

        self.model = MLP(m_features=m_features, Layer_outs=[32, 32, len(word_to_id)], activation_fn='relu', loss_type='cross_entropy')

    def _encoder(self, X):
        if not isinstance(X, Tensor):
            X = Tensor(X)
        self.X = X

        n_sentences, p_words, m_features = np.shape(X.data)

        P_E = sinusoidal_positional_encoding(p_words, m_features)
        pos_aware_embed = self.X + P_E

        tensors = [head(pos_aware_embed) for head in self.all_attention_heads]
        multi_head_attention = Tensor.cat(tensors, axis=-1) * self.W_multi_head
        add = multi_head_attention + pos_aware_embed

        eps = 1e-5
        mean = add.mean(axis=-1, keepdims=True)
        var = ((add + mean.elementwise_mul(-1)) ** 2).mean(axis=-1, keepdims=True)
        std = (var + eps) ** 0.5
        centered = add + mean.elementwise_mul(-1)
        add_n_norm = centered.elementwise_mul(std ** -1)

        return add_n_norm

    def Fit(self, X: list, Y: list, epoch=5, batch_size=1, lr=0.1, loss_type='cross_entropy'):
        n = len(X)
        epoch_loss_history = []

        for _ in range(epoch):
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for current_batch in range(0, n, batch_size):
                batch_indices = indices[current_batch:current_batch + batch_size]
                x_batch = [id_to_embed(X[i]) for i in batch_indices]
                y_batch = [Y[i] for i in batch_indices]

                y_batch = (
                    y_batch
                    if isinstance(y_batch, Tensor)
                    else Tensor(np.asarray(y_batch, dtype=float), label="Output")
                )

                pos_aware_embed = self._encoder(x_batch)
                ypred = self.model.predict(pos_aware_embed)
                # print("pos_aware_embed",pos_aware_embed, ypred)

                if y_batch.data.shape != ypred.data.shape and loss_type == 'mse':
                    raise ValueError(
                        "Prediction and target shape mismatch: "
                        f"y.shape={y_batch.data.shape}, "
                        f"ypred.shape={ypred.data.shape}"
                    )

                loss = self._GradientDescent(
                    y=y_batch,
                    ypred=ypred,
                    lr=lr,
                    loss_type=loss_type
                )
                epoch_loss += float(np.asarray(loss.data).mean())
                num_batches += 1

            epoch_loss_history.append(epoch_loss / num_batches)
        print(epoch_loss_history)
        return epoch_loss_history, loss

    def parameters(self):
        params = [self.W_multi_head]
        for head in self.all_attention_heads:
            params.extend(head.parameters())
        params.extend(self.model.parameters())
        return params
