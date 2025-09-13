#include "tensoris/tensor.hpp"

namespace tensoris {

	void set_random_seed(unsigned int seed) {
		srand(seed);
	}

	TensorFloat::TensorFloat(size_t rows, size_t cols, float value)
		: rows_(rows), cols_(cols), data_(rows *cols, value) {
	}

	float &TensorFloat::operator()(size_t i, size_t j) {
		return data_[i * cols_ + j];
	}

	const float &TensorFloat::operator()(size_t i, size_t j) const {
		return data_[i * cols_ + j];
	}

	void TensorFloat::relu_inplace() {
		for (size_t i = 0; i < data_.size(); ++i) {
			data_[i] *= data_[i] > 0 ? 1 : 0;
		}
	}

	void TensorFloat::print() {
		for (size_t i = 0; i < rows_; ++i) {
			for (size_t j = 0; j < cols_; ++j) {
				std::cout << (*this)(i, j) << " ";
			}
			std::cout << std::endl;
		}
	}

	TensorFloat matmul(const TensorFloat &A, const TensorFloat &B) {
		assert(A.cols() == B.rows());
		if (A.cols() != B.rows()) {
			throw std::invalid_argument("matmul: shape mismatch");
		}
		TensorFloat result(A.rows(), B.cols(), 0);
		
		for (size_t i = 0; i < A.rows(); ++i) {
			for (size_t k = 0; k < A.cols(); ++k) {
				const float A_ik = A(i, k);
				for (size_t j = 0; j < B.rows(); ++j) {
					result(i, j) += A_ik * B(k, j);
				}
			}
		}
		return result;
	}

	TensorFloat relu(const TensorFloat &A) {
		TensorFloat result(A.rows(), A.cols());
		size_t row_idx = 0;
		size_t col_idx = 0;
		for (float x : A) {
			result(row_idx, col_idx) = x < 0 ? 0 : x;
			++col_idx;
			if (col_idx == A.cols()) {
				col_idx = 0;
				++row_idx;
			}
		}
		return result;
	}

	TensorFloat add(const TensorFloat &A, const TensorFloat &B) {
		assert(A.rows() == B.rows());
		assert(A.cols() == B.cols());
		if ((A.rows() != B.rows()) || (A.cols() != B.cols())) {
			throw std::invalid_argument("add: shape mismatch");
		}
		TensorFloat result(A.rows(), A.cols());
		for (size_t i = 0; i < A.rows(); ++i) {
			for (size_t j = 0; j < A.cols(); ++j) {
				result(i, j) += A(i, j) + B(i, j);
			}
		}
		return result;
	}

	TensorFloat tensor_float_random_uniform(size_t rows, size_t cols, float min_value, float max_value) {
		assert(min_value < max_value);
		if (min_value >= max_value) {
			throw std::invalid_argument("tensor_float_random_uniform: min_value >= max_value");
		}
		TensorFloat result(rows, cols);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				float random_float = rand() / (float)RAND_MAX;
				result(i, j) = min_value + (max_value - min_value) * random_float;
			}
		}
		return result;
	}
}