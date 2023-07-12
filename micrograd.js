class Value {
  constructor(data, params = {}) {
    this.data = data;
    this.grad = 0.0;  // derivative of output with regards to this value
    this._backward = () => {};
    this._prev = (params._children) ? params._children : [];
    this._op = (params._op) ? params._op : '';
    this.label = (params.label) ? params.label : ''
  }

  add(other) {
    other = (other instanceof Value) ? other : new Value(other);
    let out = new Value(this.data + other.data, { _children: [this, other], _op: '+' });
    out._backward = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    other = (other instanceof Value) ? other : new Value(other);
    let out = new Value(this.data * other.data, { _children: [this, other], _op: '*' });
    out._backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  tanh() {
    let x = this.data;
    let t = (Math.exp(2*x) - 1)/(Math.exp(2*x) + 1);
    let out = new Value(t, { _children: [this], _op: 'tanh' });
    out._backward = () => {
      this.grad += (1 - t**2) * out.grad;
    };
    return out;
  }

  backward() {
    let topo = [];
    let visited = [];

    let build = (v) => {
      if (!visited.includes(v)) {
        visited.push(v);
        for (let child of v._prev) {
          build(child);
        }
        topo.push(v);
      }
    };
    build(this);

    this.grad = 1.0;
    for (let node of topo.toReversed()) {
      node._backward();
    }
  }

  exp() {
    let x = this.data;
    let out = new Value(Math.exp(x), { _children: [this], _op: 'exp' });
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  pow(other) {
    if (typeof other !== 'number') {
      throw 'Only numeric arguments supported for pow()';
    }
    let out = new Value(this.data**other, { _children: [this], _op: '**'+other });
    out._backward = () => {
      this.grad += (other * (this.data ** (other - 1))) * out.grad;
    };
    return out;
  }

  relu() {  // unused
    let out = new Value((this.data > 0) ? this.data : 0, { _children: [this], _op: 'ReLU' });
    out._backward = () => {
      this.grad += (out.data > 0) * out.grad;
    };
    return out;
  }

  // Composite operations

  div(other) {  // unused
    other = (other instanceof Value) ? other : new Value(other);
    return other.pow(-1).mul(this);
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    other = (other instanceof Value) ? other : new Value(other);
    return this.add(other.neg());
  }
}


function draw_dot(root) {
  let nodes = [];
  let edges = [];

  let build = (v) => {
    if (!nodes.includes(v)) {
      nodes.push(v);
    }
    for (let child of v._prev) {
      edges.push([child, v]);
      build(child);
    }
  };
  build(root);

  let dot = 'strict digraph { rankdir="LR" ';
  for (let i=0; i < nodes.length; i++) {
    if (nodes[i]._op) {
      dot += 'op' + i + '[label="' + nodes[i]._op + '"]; ';
    }
    let label = (nodes[i].label) ? (nodes[i].label + ' | ') : '';
    label += 'data ' + nodes[i].data + ' | grad ' + nodes[i].grad;
    dot += 'value' + i + ' [shape=record label="' +  label + '"]; ';
  }
  for (let i=0; i < nodes.length; i++) {
    if (nodes[i]._op) {
      dot += 'op' + i + ' -> value' + i + '; ';
    }
  }
  for (let i=0; i < edges.length; i++) {
    let from = edges[i][0];
    let to = edges[i][1];
    if (to._op) {
      dot += 'value' + nodes.indexOf(from) + ' -> op' + nodes.indexOf(to) + '; ';
    } else {
      dot += 'value' + nodes.indexOf(from) + ' -> value' + nodes.indexOf(to) + '; ';
    }
  }
  dot += '}';
  return dot;
}


class Neuron {
  // nin .. number of inputs
  constructor(nin) {
    this.w = [];  // array of weights for each input
    for (let i=0; i < nin; i++) {
      this.w[i] = new Value(Math.random() * 2 - 1, { label: 'w'+i });
    }
    this.b = new Value(Math.random() * 2 - 1, { label: 'b' });  // bias
  }

  forward(x) {
    if (x.length !== this.w.length) {
      throw 'forward() expected ' + this.w.length + ', ' + x.length + ' given';
    }
    // w * x + b (for each input)
    let act = this.w[0].mul(x[0]);
    for (let i=1; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    act = act.add(this.b);
    let out = act.tanh();
    return out;
  }

  parameters() {
    return [...this.w, this.b];
  }
}


class Layer {
  // nin .. number of inputs
  // nout .. number of neurons (outputs of the layer)
  constructor(nin, nout) {
    this.neurons = [];  // array of neurons
    for (let i=0; i < nout; i++) {
      this.neurons[i] = new Neuron(nin);
    }
  }

  forward(x) {
    let outs = [];
    for (let i in this.neurons) {
      outs[i] = this.neurons[i].forward(x);
    }
    return (outs.length == 1) ? outs[0] : outs;
  }

  parameters() {
    let params = [];
    for (let neuron of this.neurons) {
      params = params.concat(neuron.parameters());
    }
    return params;
  }
}


class MLP {
  // nin .. number of inputs
  // nouts ... array of neurons per layer
  constructor(nin, nouts) {
    let sz = [nin, ...nouts];  // add inputs as the first layer
    this.layers = [];
    for (let i=0; i < nouts.length; i++) {
      this.layers[i] = new Layer(sz[i], sz[i+1]);
    }
  }

  forward(x) {
    for (let layer of this.layers) {
      x = layer.forward(x)
    }
    return x;
  }

  parameters() {
    let params = [];
    for (let layer of this.layers) {
      params = params.concat(layer.parameters());
    }
    return params;
  }
}


function mean_squared_error(ygt, yout) {
  // ygt .. expected values (ground truth)
  // yout .. predicted values (Value instances)
  if (ygt.length !== yout.length) {
    throw 'Lengths of arguments don\'t match';
  }

  let loss = yout[0].sub(ygt[0]).pow(2);
  for (let i=1; i < yout.length; i++) {
    loss = loss.add(yout[i].sub(ygt[i]).pow(2));
  }
  return loss;
}
