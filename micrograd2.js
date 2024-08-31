class Value {
  constructor(data, params = {}) {
    this.data = data;
    this.grad = 0.0; // derivative of output with regards to this value
    this._backward = () => {};
    this._prev = params._children ? params._children : []; // values that are inputs to this value
    this._op = params._op ? params._op : ""; // operation that produced this value
    this.label = params.label ? params.label : ""; // displayed name
  }

  // The essence of an operation function: generates an output node and stores the elements and operation that generates it
  add(other) {
    other = other instanceof Value ? other : new Value(other);
    let out = new Value(this.data + other.data, {
      _children: [this, other],
      _op: "+",
    });
    out._backward = () => {
      // Arrow functions do not have their own context
      // Since out._backward is defined inside the add method, 'this' within _backward refers to the instance of Value on which add was originally called.

      // For every output node, calculate out.grad * dout/dn and add it to the accumulated gradient of n
      this.grad += out.grad; // dout/dn = 1 since [out = n + other]
      other.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    other = other instanceof Value ? other : new Value(other);
    let out = new Value(this.data * other.data, {
      _children: [this, other],
      _op: "*",
    });
    out._backward = () => {
      this.grad += other.data * out.grad; // dout/dn = other since [out = n * other]
      other.grad += this.data * out.grad;
    };
    return out;
  }

  tanh() {
    // Formula:
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    // tanh'(x) = dt/dx = 1 - t^2
    let x = this.data;
    let t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1);
    let out = new Value(t, { _children: [this], _op: "tanh" });
    out._backward = () => {
      this.grad += (1 - t ** 2) * out.grad;
    };
    return out;
  }

  exp() {
    let x = this.data;
    let out = new Value(Math.exp(x), { _children: [this], _op: "exp" });
    out._backward = () => {
      this.grad += out.data * out.grad; // dout/dn = out since [out = e^n]
    };
    return out;
  }

  pow(other) {
    if (typeof other !== "number") {
      throw "Only numeric arguments supported for pow()";
    }
    let out = new Value(this.data ** other, {
      _children: [this],
      _op: "**" + other,
    });
    out._backward = () => {
      this.grad += other * this.data ** (other - 1) * out.grad; // dout/dn = other * n^(other-1) since [out = n^other]
    };
    return out;
  }

  relu() {
    // unused
    let out = new Value(this.data > 0 ? this.data : 0, {
      _children: [this],
      _op: "ReLU",
    });
    out._backward = () => {
      this.grad += (out.data > 0) * out.grad;
    };
    return out;
  }

  // Composite operations

  div(other) {
    // unused
    other = other instanceof Value ? other : new Value(other);
    return other.pow(-1).mul(this);
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    other = other instanceof Value ? other : new Value(other);
    return this.add(other.neg());
  }

  // backpropagate the network starting from this node
  backward() {
    // list all previous nodes in topological order
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

    this.grad = 1.0; // Typically, this final node is the loss in a neural network
    for (let node of topo.toReversed()) {
      node._backward();
    }
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

  // generate string in DOT language -- see README
  let dot = 'strict digraph { rankdir="LR" ';
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i]._op) {
      dot += "op" + i + '[label="' + nodes[i]._op + '"]; ';
    }
    let label = nodes[i].label ? nodes[i].label + " | " : "";
    label += "data " + nodes[i].data + " | grad " + nodes[i].grad;
    dot += "value" + i + ' [shape=record label="' + label + '"]; ';
  }
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i]._op) {
      dot += "op" + i + " -> value" + i + "; ";
    }
  }
  for (let i = 0; i < edges.length; i++) {
    let from = edges[i][0];
    let to = edges[i][1];
    if (to._op) {
      dot +=
        "value" + nodes.indexOf(from) + " -> op" + nodes.indexOf(to) + "; ";
    } else {
      dot +=
        "value" + nodes.indexOf(from) + " -> value" + nodes.indexOf(to) + "; ";
    }
  }
  dot += "}";
  return dot;
}

class Neuron {
  // nin .. number of inputs
  constructor(nin, layerIndex, neuronIndex) {
    this.layerIndex = layerIndex;
    this.neuronIndex = neuronIndex;
    this.w = []; // array of weights for each input
    for (let i = 0; i < nin; i++) {
      this.w[i] = new Value(Math.random() * 2 - 1, { label: "w" + i });
    }
    this.b = new Value(Math.random() * 2 - 1, { label: "b" }); // bias
    this.out = new Value(0.0); // output of the neuron
  }

  forward(x) {
    // x .. array of inputs
    if (x.length !== this.w.length) {
      throw "forward() expected " + this.w.length + ", " + x.length + " given";
    }
    // w * x + b (for each input)
    let act = this.w[0].mul(x[0]);
    for (let i = 1; i < this.w.length; i++) {
      act = act.add(this.w[i].mul(x[i]));
    }
    act = act.add(this.b);
    let out = act.tanh();
    this.out = out;
    return out;
  }

  parameters() {
    return [...this.w, this.b];
  }

  printParams() {
    // console.log(
    //   "w: ",
    //   this.w.map((w) => w.data.toFixed(2)),
    //   "b: ",
    //   this.b.data
    // );
    // print a string representation of the neuron
    let w = this.w.map((w) => w.data.toFixed(2));
    let b = this.b.data.toFixed(2);
    let result = `l${this.layerIndex}n${this.neuronIndex}: `;
    for (let i in w) {
      result += w[i];
      result += " ";
    }
    result += `${b}\nGrad: `;
    // print gradients of weights and bias
    for (let i in this.w) {
      result += `${this.w[i].grad.toFixed(2)} `;
    }
    result += `${this.b.grad.toFixed(2)}`;
    console.log(result);
  }

  printId() {
    console.log(`l${this.layerIndex}n${this.neuronIndex}`);
  }
}

class Layer {
  // nin .. number of inputs
  // nout .. number of neurons (outputs of the layer)
  constructor(nin, nout, layerIndex) {
    this.nin = nin;
    this.nout = nout;
    this.neurons = []; // array of neurons
    for (let i = 0; i < nout; i++) {
      this.neurons[i] = new Neuron(nin, layerIndex, i);
    }
  }

  forward(x) {
    let outs = [];
    for (let i in this.neurons) {
      outs[i] = this.neurons[i].forward(x);
    }
    return outs.length == 1 ? outs[0] : outs;
  }

  parameters() {
    let params = [];
    for (let neuron of this.neurons) {
      params = params.concat(neuron.parameters());
    }
    return params;
  }

  printParams() {
    for (let neuron of this.neurons) {
      neuron.printParams();
    }
  }
}

class MLP {
  // nin .. number of inputs
  // nouts ... array of neurons per layer
  constructor(nin, nouts) {
    this.sz = [nin, ...nouts]; // add inputs as the first layer
    this.layers = []; // not including the input layer
    for (let i = 0; i < nouts.length; i++) {
      this.layers[i] = new Layer(this.sz[i], this.sz[i + 1], i);
    }
  }

  forward(x) {
    for (let layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  // function for adding a new node to a specific layer
  add_node(layer, neuron) {
    this.layers[layer].neurons.push(neuron);
    // add a new weight for each node in the next layer
    let next_layer = this.layers[layer + 1];
    for (let i = 0; i < next_layer.neurons.length; i++) {
      next_layer.neurons[i].w.push(new Value(Math.random() * 2 - 1));
    }
    // update the size of the network
    this.sz[layer + 1] += 1;
  }

  add_layer(idx, nout) {
    let last_layer = this.layers[idx - 2];
    let next_layer = this.layers[idx - 1];
    let new_layer = new Layer(last_layer.nout, nout);
    this.layers.splice(idx - 1, 0, new_layer);
    this.sz.splice(idx, 0, nout);
    // reset the weights of the next layer
    for (let i = 0; i < next_layer.neurons.length; i++) {
      next_layer.neurons[i].w = [];
      for (let j = 0; j < nout; j++) {
        next_layer.neurons[i].w[j] = new Value(Math.random() * 2 - 1);
      }
    }
  }

  parameters() {
    let params = [];
    for (let layer of this.layers) {
      params = params.concat(layer.parameters());
    }
    return params;
  }

  printParams() {
    for (let layer of this.layers) {
      layer.printParams();
    }
  }

  printSize() {
    console.log(
      "netsize: \n",
      "this.layers: ",
      this.layers.map((layer) => layer.neurons.length),
      "\nthis.size",
      this.sz
    );
  }
}

function mean_squared_error(ygt, yout) {
  // ygt .. expected values (ground truth)
  // yout .. predicted values (Value instances)
  if (ygt.length !== yout.length) {
    throw "Lengths of arguments don't match";
  }

  let loss = yout[0].sub(ygt[0]).pow(2);
  for (let i = 1; i < yout.length; i++) {
    loss = loss.add(yout[i].sub(ygt[i]).pow(2));
  }
  console.log("loss: ", loss);
  return loss;
}

// export { Value, Neuron, Layer, MLP, mean_squared_error, draw_dot };
