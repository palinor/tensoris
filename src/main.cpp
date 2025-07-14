#include "tensoris/tensor.hpp"

int main() {
	tensoris::TensorFloat test(5, 5, 0.3);
	tensoris::TensorFloat test2(5, 5, -2);

	tensoris::TensorFloat result = tensoris::add(test, test2);
	tensoris::TensorFloat result2 = tensoris::relu(result);
	result.print();
	result2.print();
}