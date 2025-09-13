#include <chrono>
#include "tensoris/tensor.hpp"
#include "tensoris/profiler.hpp"

using Clock = std::chrono::steady_clock;

struct Case {
	std::string name;
	size_t rows, cols;
	int64_t repeats;
	int64_t warmup;
};

static std::vector<Case> default_small_cases() {
	return {
		{"tiny-64x64", 64, 64, 2, 2},
		{"small-128x128", 128, 128, 1, 1},
		{"mid-256x256", 256, 256, 4, 5},
		{"big-512x512", 512, 512, 1, 2},
		{"xl-1024x1024", 1024, 1024, 1, 1}
	};
}

static std::vector<Case> default_cases() {
	return {
		{"tiny-64x64", 64, 64, 2000, 200},
		{"small-128x128", 128, 128, 1000, 100},
		{"mid-256x256", 256, 256, 400, 50},
		{"big-512x512", 512, 512, 150, 20},
		{"xl-1024x1024", 1024, 1024, 40, 6}
	};
}

void test_relu() {
	tensoris::TensorFloat test(5, 5, 0.3);
	tensoris::TensorFloat test2(5, 5, -2);

	tensoris::TensorFloat result = tensoris::add(test, test2);
	tensoris::TensorFloat result2 = tensoris::relu(result);
	result.print();
	result2.print();
}

void test_profiler() {
	tensoris_profile::Config config{};
	// config.enabled = std::getenv("TENSORIS_PROFILER") != nullptr;
	config.enabled = true;
	config.logfile_path = "C:\\Users\\aionf\\Github\\tensoris\\build\\prof\\trace.json";
	tensoris_profile::init_config(config);

	auto cases = default_cases();
	uint64_t global_seed = 42;

	{
		TPROF_SCOPE("profile_test_run");

		for (const auto &this_case : cases) {
			{
				TPROF_SCOPE(this_case.name.c_str());
				auto A = tensoris::tensor_float_random_uniform(
					this_case.rows,
					this_case.cols
				);
				auto B = tensoris::tensor_float_random_uniform(
					this_case.rows,
					this_case.cols
				);
				auto C = tensoris::tensor_float_random_uniform(
					this_case.rows,
					this_case.cols
				);

				tensoris::TensorFloat temp_tensor_1({ this_case.rows, this_case.cols });
				tensoris::TensorFloat temp_tensor_2({ this_case.rows, this_case.cols });
				tensoris::TensorFloat out({ this_case.rows, this_case.cols });

				const int warmup = this_case.warmup;
				const int repeats = this_case.repeats;

				for (int i = 0; i < warmup; ++i) {
					{ TPROF_SCOPE("add"); temp_tensor_1 = tensoris::add(A, B); }
					{ TPROF_SCOPE("relu"); temp_tensor_2 = tensoris::relu(temp_tensor_1); }
					{ TPROF_SCOPE("multiply"); out = tensoris::matmul(temp_tensor_2, C); }
				}

				auto t_0 = Clock::now();

				for (int i = 0; i < repeats; ++i) {
					{ TPROF_SCOPE("add"); temp_tensor_1 = tensoris::add(A, B); }
					{ TPROF_SCOPE("relu"); temp_tensor_2 = tensoris::relu(temp_tensor_1); }
					{ TPROF_SCOPE("multiply"); out = tensoris::matmul(temp_tensor_2, C); }
					if ((i & 0x3F) == 0) {
						tensoris_profile::event_counter(tensoris_profile::intern("iter"), i);
					}
				}
				auto t_1 = Clock::now();

				const double milliseconds = std::chrono::duration<double, std::milli>(t_1 - t_0).count();
				tensoris_profile::event_mark(tensoris_profile::intern("case_done"));
				tensoris_profile::event_counter(tensoris_profile::intern((this_case.name + "/latency_ms").c_str()), (int64_t)milliseconds);

				std::cout << this_case.name << ": " << milliseconds << " ms for " << repeats
					<< " iters (" << (milliseconds / repeats) << " ms/iter)\n";

			}
		}
	}
	tensoris_profile::shutdown();
}



int main() {
	test_profiler();
}