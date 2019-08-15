#include "compiler.h"

#include <stack>

using namespace torch::jit;


bool TestCompiler::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::relu:
      return true;
    default:
      return false;
  }
  return false;
}


void TestCompiler::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<IValue> inputs = last(stack, num_inputs);


  // Run the compiled function
  auto outputs = cache_[spec](inputs);

  drop(stack, num_inputs);
  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(IValue(var));
  }
}

void TestCompiler::emitOperation(
    const Node* node,
    const std::set<const Node*>& seen,) {
  switch (node->kind()) {
    case aten::relu: {
    	// Operation to do 
    }
  }

}
