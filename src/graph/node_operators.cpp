#include "node_operators.h"
#include "expression_graph.h"

#include "tensors/tensor_operators.h"

namespace marian {

size_t ConstantNode::allocate() {
  size_t elements = 0;
  if(!val_) {
    graph()->allocateForward(this);
    elements = val_->shape().elements();
  }
  return elements;
}

void ConstantNode::init() {
  if(!initialized_) {
    init_->apply(val_);
    initialized_ = true;
  }
  init_.reset();
}

ParamNode::ParamNode(Ptr<ExpressionGraph> graph,
                     const Shape& shape,
                     const Ptr<inits::NodeInitializer>& init,
                     bool fixed)
    : ParamNode(graph, shape, init, Type::float32, fixed) {}

ParamNode::ParamNode(Ptr<ExpressionGraph> graph,
                     const Shape& shape,
                     const Ptr<inits::NodeInitializer>& init,
                     const Type& valueType,
                     bool fixed)
    : Node(graph, shape, valueType),
      init_(init),
      initialized_(false) {
  init_->setGraph(graph);
  setTrainable(!fixed);
  setMemoize(graph->isInference());
}

void ParamNode::init() {
  if(!initialized_) {
    init_->apply(val_);
    initialized_ = true;
  }
  init_.reset();
}
}  // namespace marian
