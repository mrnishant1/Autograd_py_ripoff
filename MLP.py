from typing import Literal
import random
import warnings

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, m_features, non_linear=True,activation_fn: Literal["tanh", "relu"] = "relu"):
        std = np.sqrt(2 / m_features)       
        self.W = Tensor(np.random.randn(m_features,1) * std,label="w")
        self.b = Tensor([[0.0]],label="bias")
        # print("Weights: ",self.W)

        self.non_linear = non_linear
        self.activation_fn = activation_fn

    def __call__(self, x):

        # print("lr:", lr)
        # print("W:", self.W.data.min(), self.W.data.max())
        # print("grad:", self.W.grad.min(), self.W.grad.max())
        # x is expected to be:
        # (batch_size, m_features)
        x = Tensor(np.asarray(x, dtype=float)) if not isinstance(x, Tensor) else x

        # for matmul (x,W) == (n_inp, p_words, m_features),(m_features,1) --> (n_inp,p_words,1)
        
        # print(((x.data.shape),self.W.data.shape))
        act = ( x * self.W ) + self.b

        # Return matmul shape: (n_inp, p_words, 1)
        # Final layer: return raw logits
        if not self.non_linear:
            return act

        # Hidden layer activation
        activation = getattr(act, self.activation_fn)()
        return activation

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        activation = self.activation_fn if self.non_linear else "linear"
        return f"{activation} Neuron({self.W.data.shape[1]})"


class Layer(Module):
    def __init__(self,layer_id,m_features,n_out,activation_fn="relu",non_linear=True):
        self.layer_id = layer_id
        self.non_linear = non_linear
        self.neurons = [Neuron(m_features=m_features,activation_fn=activation_fn,non_linear=non_linear) for _ in range(n_out)]

    def __call__(self, x):
        # Each neuron produces:(n_inp,p_words,1) * L neuron .Cat --> (n_inp,p_words, L )
        outputs = [neuron(x) for neuron in self.neurons]
        outputs:Tensor = Tensor.cat(outputs, axis=-1)
        # print("input", outputs)
        if not self.non_linear:
            if outputs.data.shape[-1] > 1:
                outputs = outputs.softmax()
                # print("output", outputs)
        # print(f"Layer id: {self.layer_id},outputs.data.min(),outputs.data.max()")
        return outputs

    def parameters(self):
        return [ p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return (f"Layer of " f"[{', '.join(str(n) for n in self.neurons)}]")


class MLP(Module):

    def __init__( self, m_features: int, Layer_outs: list[int], activation_fn="relu",loss_type: Literal["mse", "cross_entropy"] = "mse"):
        input_size = [m_features] + Layer_outs
        self.loss_type = loss_type
        self.layers = [
            Layer(
                layer_id=i,
                m_features=input_size[i],
                n_out=input_size[i + 1],

                # Hidden layers -> activation
                # Final layer -> linear
                non_linear=(i != len(Layer_outs) - 1),
                activation_fn=activation_fn
            )
            for i in range(len(Layer_outs))
        ]

    def predict(self, x):
        # Convert input to Tensor once
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=float))

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [ p for layer in self.layers for p in layer.parameters()]

    def Fit( self, x, y, epoch=200, lr=0.0001, batch_size=32):
        n = len(x)
        epoch_loss_history = []

        for _ in range(epoch):

            # Shuffle every epoch
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for current_batch in range(0, n, batch_size):

                batch_indices = indices[current_batch:current_batch + batch_size]
                # x_batch = [id_to_embed(x[i]) for i in batch_indices]
                x_batch = [x[i] for i in batch_indices]
                y_batch = [y[i] for i in batch_indices]
                # Forward pass
                ypred = self.predict(x_batch)
               
                # print(f"Epoch: {epoch_no}, batch Numer: {current_batch}: yprediction: {ypred.data.min,ypred.data.max}", )
                # Convert target to Tensor
                y_batch = (
                    y_batch
                    if isinstance(y_batch, Tensor)
                    else Tensor(np.asarray(y_batch, dtype=float), label="Output")
                )

                # Check shape
                if y_batch.data.shape != ypred.data.shape and self.loss_type =='mse':
                    raise ValueError(
                        "Prediction and target shape mismatch: "
                        f"y.shape={y_batch.data.shape}, "
                        f"ypred.shape={ypred.data.shape}"
                    )

                # Forward + backward + update
                loss = self._GradientDescent(
                    y=y_batch, ## Lables [-100,id,-100,---]
                    ypred=ypred, 
                    lr=lr,
                    loss_type=self.loss_type
                )
                # Store scalar loss
                epoch_loss += float(np.asarray(loss.data).mean())
                num_batches += 1

            # Average batch losses
            
            epoch_loss_history.append(epoch_loss / num_batches)
        print(epoch_loss_history)
        return epoch_loss_history

    def _GradientDescent(self,y, ypred, loss_type, lr=0.1):

        # Co  nvert y to Tensor
        if not isinstance(y, Tensor):
            y = Tensor(
                np.asarray(y, dtype=float),
                label="Output"
            )

        # Reset gradients
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

        # Calculate loss
        loss:Tensor = self._Calculate_Loss(
            ypred=ypred, 
            y=y,  ## Lables [-100,id,-100,---]
            loss_type=loss_type,
        )

        # Backprop
        loss.backward()
        # print("lossB", loss.grad)
        
        # Gradient descent
        for p in self.parameters():
            p.data -= lr * p.grad

        return loss

    def _Calculate_Loss(self,ypred: Tensor, y: Tensor, loss_type,):
        # y: Labels = [-100,-100,...id...]
        # ypred: vocab probs 
        # x: Actaul Ids with masked

        if loss_type == "mse":

            # Element-wise squared error
            squared_error = (ypred - y) ** 2
            # print(ypred)
            # Mean over all elements
            return squared_error.mean()

        elif loss_type == "cross_entropy":
            safe_labels = np.where(np.asarray(y.data) == -100, 0, np.asarray(y.data)).astype(int)
            batch_size, seq_len, vocab_size = ypred.data.shape
            assert vocab_size == len(vocabulary), f"vocab_size mismatch: got {vocab_size}, expected {len(vocabulary)}"

            batch_idx = np.arange(batch_size)[:, None] * np.ones((1, seq_len), dtype=int)
            seq_idx = np.arange(seq_len)[None, :] * np.ones((batch_size, 1), dtype=int)
            mask = Tensor((np.asarray(y.data) != -100).astype(float))  # (batch, seq)
            number_of_valid = mask.sum()

            eps = Tensor([[1e-7]])
            safe_ypred = ypred + eps
            picked = safe_ypred.pick(batch_idx, seq_idx, safe_labels)  # (batch, seq)

            log_probs = picked.log()                                   # (batch, seq) -- already per-word, no sum needed
            masked_loss = log_probs.elementwise_mul(mask)               # (batch, seq)
            loss = masked_loss.sum().elementwise_mul(-1.0).elementwise_mul(number_of_valid ** -1)
            return loss
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )
            
    def __repr__(self):
        return (
            f"MLP of "
            f"[{', '.join(str(layer) for layer in self.layers)}]"
        )
        
