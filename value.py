import math

class value:
    def __init__(self,data,_children = (),_op = ''):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._op = _op
        self._backward = lambda : None

    def __add__(self,other):
        other = other if isinstance(other,value)else value(other)
        out = value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __radd__(self,other):
        return self + other
    def __mul__(self,other):
        other = other if isinstance(other,value)else value(other)
        out = value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self,other):
        return self * other
    def __sub__(self,other):
        return self + (-other)
    def tanh(self):
        x = self.data
        t = ((math.exp(2*x) - 1) / (math.exp(2*x) +1))
        out = value(t,(self,),f"tanh()")
        def _backward():
            self.grad += (1 - (t**2)) * out.grad
        out._backward = _backward
        return out
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = value(self.data ** other,(self,),f"pow()")
        def _backward():
            self.grad += other * self.data **(other - 1) * out.grad
        out._backward = _backward    
        return out 
    def __truediv__(self,other):
        return self * other** -1
    def __rtruediv__(self,other):
        return other * self ** -1
    def __neg__(self):
        return self * -1
    def exp(self):
        x = self.data
        out = value(math.exp(x),(self,),f"exp()")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out 
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    def __repr__(self):
        return f"(value (data =  {self.data},grad  = {self.grad})"        


    
            
        

    
   
