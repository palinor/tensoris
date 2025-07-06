#pragma once

#include <vector>
#include <iostream>


namespace tensoris {
	
	class Tensor {
	public:

		Tensor(size_t rows, size_t cols, float value = 0.0f);


		float &operator()(size_t i, size_t j);
		const float &operator()(size_t i, size_t j) const;

		size_t rows() const { return m_rows; }
		size_t cols() const { return m_cols; }
		void print();

	private:
		size_t m_rows;
		size_t m_cols;
		std::vector<float> m_data;
	};
}