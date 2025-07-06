#include "tensoris/tensor.hpp"

namespace tensoris {
	Tensor::Tensor(size_t rows, size_t cols, float value)
		: m_rows(rows), m_cols(cols), m_data(rows *cols, value) {
	}

	float &Tensor::operator()(size_t i, size_t j) {
		return m_data[i * m_cols + j];
	}

	const float &Tensor::operator()(size_t i, size_t j) const {
		return m_data[i * m_cols + j];
	}

	void Tensor::print() {
		for (size_t i = 0; i < m_rows; ++i) {
			for (size_t j = 0; j < m_cols; ++j) {
				std::cout << (*this)(i, j) << " ";
			}
			std::cout << std::endl;
		}
	}

}