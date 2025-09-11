#include "tensoris/profiler.hpp"
#include <mutex>
#include <unordered_map>
#include <thread>
#include <fstream>
#include <chrono>

namespace tensoris_profile {
	enum class Phase: uint8_t {
		Begin,
		End,
		Counter,
		Mark
	};

	struct Event {
		uint64_t timestamp;
		uint32_t thread_id;
		tensoris_profile::NameId id;
		Phase phase;
		int64_t value;
	};

	namespace {
		std::atomic<bool> g_enabled{false};
		Config g_config;
		std::mutex g_mutex;

		struct NameTable {
			std::mutex mutex;
			std::vector<std::string> names;
			std::unordered_map<std::string_view, NameId> idx;
		} g_names;

		uint64_t epoch_nanoseconds = 0;

		uint64_t monotonic_nanoseconds() noexcept {
			using clock = std::chrono::steady_clock;
			return uint64_t (
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					clock::now().time_since_epoch()
				).count()
			) - epoch_nanoseconds;
		}

		uint32_t logical_tid() noexcept {
			static thread_local uint32_t tid = [] {
				static std::atomic<uint32_t> next{1};
				return next.fetch_add(1, std::memory_order_relaxed);
			}();
			return tid;
		};

		struct Recorder {
			Event *event_buffer = nullptr;
			size_t capacity;
			std::atomic<size_t> head{0};
			size_t tail = 0;
			bool dropped_events = false;

			void write(const Event& e) noexcept {
				if (!g_enabled.load(std::memory_order_relaxed)) return;
				auto h = head.load(std::memory_order_relaxed);
				auto next = h + 1;
				if (next - tail > capacity) {
					tail = next - capacity;
					dropped_events = true;
				}
				event_buffer[h % capacity] = e;
				head.store(next, std::memory_order_relaxed);
			}
		};
		thread_local Recorder recorder;
		std::vector<Recorder*> g_all_recorders;
	}

	void init_config(const Config& config) {
		std::lock_guard<std::mutex> lock(g_mutex);
		g_config = config;
		epoch_nanoseconds = monotonic_nanoseconds();
		if (recorder.event_buffer == nullptr) {
			recorder.capacity = config.per_thread_capacity;
			recorder.event_buffer = new Event[recorder.capacity];
			g_all_recorders.push_back(&recorder);
		}
		g_enabled.store(config.enabled, std::memory_order_relaxed);
	}

	void enable(bool on) {
		g_enabled.store(on, std::memory_order_relaxed);
	}

	bool enabled() noexcept {
		return g_enabled.load(std::memory_order_relaxed);
	}

	NameId intern(std::string_view name) noexcept {
		static thread_local std::unordered_map<std::string_view, NameId> fast;
		auto it = fast.find(name);
		if (it != fast.end()) {
			return it->second;
		}
		std::lock_guard<std::mutex> lock(g_names.mutex);
		if (auto it = g_names.idx.find(name); it != g_names.idx.end()) {
			fast[name] = it->second;
			return it->second;
		}
		NameId id = (NameId)g_names.names.size();
		g_names.idx.emplace(g_names.names.back(), id);
		fast[name] = id;
		return id;
	}

	uint64_t now_in_nanoseconds() noexcept {
		return monotonic_nanoseconds();
	}

	uint32_t t_id() noexcept {
		return logical_tid();
	}

	Scope::Scope(NameId id) noexcept : id(id)
}
