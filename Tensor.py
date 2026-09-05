import numpy as np
class Tensor:
    def __init__(self, data, children=(), op='', label=''):
        self.data = np.atleast_2d(np.asarray(data))
        self.label = label
        self.grad = np.asarray(np.zeros(shape=(self.data.shape)))
        #Internal variable for in
        self._op = op
        self._prev = children
        self._backward = lambda:None

    def __repr__(self):
        return f"Tensor :{self.data}"

    def __mul__(self, other):
        try:
            if not isinstance(other, Tensor):
                other = Tensor(other)

            if other.data.shape == (1, 1) and self.data.shape[1] != other.data.shape[0]:
                return self.elementwise_mul(other)


            # print("dims Self",self.data.shape,"dims other: ", (other.data.shape))
            out = Tensor(
                data = np.matmul(self.data, other.data),
                children=(self, other),
                op='*'
            )
            def _backward():
                self.grad += np.matmul(out.grad, other.data.swapaxes(-1, -2))
                other.grad += self.unbroadcast(np.matmul(self.data.swapaxes(-1,-2), out.grad),other.data.shape)

            out._backward = _backward
            return out
        except Exception as e:
            raise ValueError(
                f"Matrix multiplication failed: self dimensions {self.data.shape}, "
                f"other dimensions {other.data.shape}"
            ) from e

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(np.pow(self.data,other), children=(self,), op=f'**{other}')
        def _backward():
            self.grad += (other * np.pow(self.data,other-1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data),children=(self,), op="exp")

        def _backward():
            self.grad += out.data * out.grad  #element wise multiplication
        out._backward = _backward
        return out

    def transpose(self):
        out = Tensor(np.array(self.data).transpose(),op='T',children=(self,))
        def _backward():
            self.grad += out.grad.transpose()
        out._backward = _backward
        return out

    def softmax(self) -> 'Tensor':
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        softmax_data = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

        out = Tensor(softmax_data, op='softmax', children=(self,))

        def _backward():
            # correct softmax Jacobian, per row:
            # dL/dx_i = s_i * (dL/dout_i - sum_j(s_j * dL/dout_j))
            s = out.data
            dot = np.sum(s * out.grad, axis=-1, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out

    def relu(self):
        b = self.data
        out = Tensor(np.where(np.greater_equal(0, b), 0, b),children=(self,),label='ReLU')

        def _backward():
            self.grad += (out.data>0)*out.grad

        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)


        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.asarray(np.ones(shape=(self.data.shape)))

        for v in reversed(topo):
            v._backward()

    def __sub__(self, other): # self - other
        return self + (-other)

    def __neg__(self): # -self
        return self * -1

    def __rmul__(self, other): #__rmul__ is used by python when a * b can't be calculated then python tries b * a
        return self*other

    def __radd__(self, other): # other + self
        return self + other

    def cat(tensors:list['Tensor'], axis = -1):
        if not isinstance(tensors, (list, tuple)):
            raise TypeError("cat() expects a list or tuple of Tensor objects")
        if len(tensors) == 0:
            raise ValueError("cat() requires at least one tensor")
        if not all(isinstance(t, Tensor) for t in tensors):
            raise TypeError("All elements in tensors must be Tensor instances")
        if len(tensors)==1: return tensors[0]

        out_data = np.concatenate([t.data for t in tensors],axis=axis)

        out = Tensor(out_data,
           children=tuple(tensors),
            op="cat"
        )
        def _backward():
            start = 0
            for t in tensors:
                size = t.grad.shape[axis]
                end = start + size
                idx = [slice(None)] * out.grad.ndim
                idx[axis] = slice(start,end)
                t.grad += out.grad[tuple(idx)]
                start = end
        out._backward = _backward
        return out

    def unbroadcast(self,grad, shape, label=''):
        # Remove extra dimensions on the left
        while grad.ndim > len(shape):
            grad = np.sum(grad, axis=0)
        # Sum dimensions where original shape was 1
        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = np.sum(grad, axis=axis, keepdims=True)
        return grad

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(np.add(self.data, other.data), children=(self, other), op='+')
        def _backward():
            # print("trying to add: ", other.grad, out.data.shape)
            self.grad += self.unbroadcast(out.grad, self.data.shape)
            other.grad += self.unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def elementwise_mul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(
            data=self.data * other.data,
            children=(self, other),
            op='ew*'
        )
        def _backward():
            self.grad +=  self.unbroadcast(other.data * out.grad, shape=self.data.shape)
            other.grad += self.unbroadcast(self.data * out.grad, shape=other.data.shape)
        out._backward = _backward
        return out

    def mean(self,axis=1,keepdims=True):
        out = Tensor(np.mean(self.data, axis=axis,keepdims=keepdims),children=(self,),op='mean')
        def _backward():
            if axis is None:
                # If mean is taken over all elements, gradient is distributed equally
                self.grad += out.grad / self.data.size
            else:
                # If mean is taken over a specific axis
                self.grad += (1/self.data.shape[axis])*out.grad
        out._backward = _backward
        return out

    def std(self,axis = 1,keepdims=True):
        out = Tensor(np.std(self.data,axis=axis,keepdims=keepdims),children=(self,),op='std')
        def _backward():
            self.grad = out.grad * (self.data - np.mean(self.data)) / (self.data.shape[axis] * out.data)
        out._backward = _backward
        return out

    def tanh(self):
        e = self.elementwise_mul(2).exp()
        out = Tensor(
            (e - 1).data * ((e + 1)**-1).data,
            children=(self,),
            op='tanh'
        )

        def _backward():
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward
        return out
    def log(self):
        out = Tensor(np.log(self.data), children=(self,), op='log')

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out
    def sum(self, axis=1,keepdims=True):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims,dtype=float), children=(self,), op='sum')
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad  # broadcast grad back to original shape
        
        out._backward = _backward
        return out
    
    def pick(self, batch_idx, seq_idx, label_idx):
        # self.data shape: (batch, seq, vocab)
        # batch_idx, seq_idx, label_idx: numpy arrays, shape (batch, seq)
        picked_data = self.data[batch_idx, seq_idx, label_idx]   # shape (batch, seq)

        out = Tensor(picked_data, children=(self,), op='pick')

        def _backward():
            grad = np.zeros_like(self.data)               # (batch, seq, vocab)
            grad[batch_idx, seq_idx, label_idx] = out.grad # scatter grad back to picked positions
            self.grad += grad

        out._backward = _backward
        return out
    def swapaxes(self, axis1, axis2):
        out = Tensor(
            np.swapaxes(self.data, axis1, axis2),
            children=(self,),
            op='swapaxes'
        )

        def _backward():
            self.grad += np.swapaxes(out.grad, axis1, axis2)

        out._backward = _backward
        return out
