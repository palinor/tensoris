#pragma once

#include<cassert>
#include<cmath>
#include <vector>
#include <iostream>


namespace tensoris {

	void set_random_seed(unsigned int seed);
	
	class TensorFloat {
	public:

		TensorFloat(size_t rows, size_t cols, float value = 0.0f);


		float &operator()(size_t i, size_t j);
		const float &operator()(size_t i, size_t j) const;

		size_t rows() const { return rows_; }
		size_t cols() const { return cols_; }
		void print();

		std::vector<float>::const_iterator begin() const {
			return data_.begin();
		}

		std::vector<float>::const_iterator end() const {
			return data_.end();
		}

		void relu_inplace();
		std::unique_ptr<TensorFloat> grad_;

		void zero_grad() {
			std::fill(grad_->data_.begin(), grad_->data_.end(), 0.0f);
		}

		TensorFloat *grad() {
			if (!grad_) {
				grad_ = std::make_unique<TensorFloat>(rows_, cols_);
			}
			return grad_.get();
		}



	private:
		size_t rows_;
		size_t cols_;
		std::vector<float> data_;
	};

	TensorFloat matmul(const TensorFloat &A, const TensorFloat &B);
	TensorFloat add(const TensorFloat &A, const TensorFloat &B);
	TensorFloat relu(const TensorFloat &A);

	TensorFloat tensor_float_random_uniform(size_t rows, size_t cols, float min_value = 0.0f, float max_value = 1.0f);

}