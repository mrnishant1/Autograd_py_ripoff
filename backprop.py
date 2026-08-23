import numpy as np
from graphviz import Digraph

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
        
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        # print("add: ", self, other)
        out = Tensor(np.add(self.data, other.data), children=(self, other), op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

 
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        print("dims Self",self.data.shape,"dims other: ", other.data.shape)
        out = Tensor(
            data=self.data @ other.data,
            children=(self, other),
            op='*'
        )
        def _backward():  
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

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
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
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
            print("Hiiiiiiiii",  self.data.shape, out.grad.shape,"hi", self.grad.shape)
            self.grad += out.data * out.grad  #element wise multiplication
        self._backward = _backward
        return out
    
    def transpose(self):
        out = Tensor(np.array(self.data).transpose(),op='T',children=(self,))
        def _backward():
            self.grad += out.grad.transpose()
        self._backward = _backward 
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

    
    def softmax(self) -> Tensor:
        shifted = self.data - np.max(self.data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        softmax_data = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        out = Tensor(softmax_data, op='softmax', children=(self,))

        def _backward():
            # correct softmax Jacobian, per row:
            # dL/dx_i = s_i * (dL/dout_i - sum_j(s_j * dL/dout_j))
            s = out.data
            dot = np.sum(s * out.grad, axis=1, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _backward
        return out
        
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        # dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        dot.node(name=str(id(n)), label = "{ label: %s | data %s |  Grad: %s }" %  (n.label, str(n.data), str(n.grad)), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot