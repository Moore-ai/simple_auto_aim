#include <cassert>

#include "tasks/auto_aim/classifier.hpp"

int main()
{
  assert(auto_aim::Classifier::uses_openvino("CPU"));
  assert(!auto_aim::Classifier::uses_openvino("GPU"));
  return 0;
}
