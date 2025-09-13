#include "tensoris/profiler.hpp"
#include <mutex>
#include <unordered_map>
#include <thread>
#include <fstream>
#include <chrono>

namespace tensoris_profile {
	enum class Phase : uint8_t {
		Begin,
		End,
		Counter,
		Mark
	};

	struct Event {
		uint64_t timestamp_in_nanoseconds;
		uint32_t thread_id;
		tensoris_profile::NameId id;
		Phase phase;
		int64_t value;
	};

	namespace {
		std::atomic<bool> g_enabled{ false };
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
			return uint64_t(
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					clock::now().time_since_epoch()
				).count()
			) - epoch_nanoseconds;
		}

		uint32_t logical_tid() noexcept {
			static thread_local uint32_t tid = [] {
				static std::atomic<uint32_t> next{ 1 };
				return next.fetch_add(1, std::memory_order_relaxed);
				}();
			return tid;
		};

		struct Recorder {
			Event *event_buffer = nullptr;
			size_t capacity;
			std::atomic<size_t> head{ 0 };
			size_t tail = 0;
			bool dropped_events = false;

			void write(const Event &e) noexcept {
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
		std::vector<Recorder *> g_all_recorders;
	}

	void init_config(const Config &config) {
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

	bool is_enabled() noexcept {
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
		g_names.names.emplace_back(name);
		g_names.idx.emplace(std::make_pair(g_names.names.back(), id));
		fast[name] = id;
		return id;
	}

	uint64_t now_in_nanoseconds() noexcept {
		return monotonic_nanoseconds();
	}

	uint32_t thread_id() noexcept {
		return logical_tid();
	}

	Scope::Scope(NameId scope_name) noexcept :
		scope_name(scope_name),
		start_ns(now_in_nanoseconds())
	{
		begin_scope(scope_name);
	}
	Scope::~Scope() noexcept {
		end_scope(scope_name);
	}

	void begin_scope(NameId scope_name) noexcept {
		if (!is_enabled()) {
			return;
		}
		recorder.write(Event{ now_in_nanoseconds(), thread_id(), scope_name, Phase::Begin, 0 });
	}

	void end_scope(NameId scope_name) noexcept {
		if (!is_enabled()) {
			return;
		}
		recorder.write(Event{ now_in_nanoseconds(), thread_id(), scope_name, Phase::End, 0 });
	}

	void event_counter(NameId scope_name, int64_t v) noexcept {
		if (!is_enabled()) {
			return;
		}
		recorder.write(Event{ now_in_nanoseconds(), thread_id(), scope_name, Phase::Counter, v });
	}

	void event_mark(NameId scope_name) noexcept {
		if (is_enabled()) {
			return;
		}
		recorder.write(Event{ now_in_nanoseconds(), thread_id(), scope_name, Phase::Mark, 0 });
	}

	void flush_thread() noexcept {}

	static void write_trace(
		std::ostream &os,
		const std::vector<Event> &events
	) {
		os << "{\"traceEvents\":[";
		bool first = true;
		auto emit = [&](const char *name, const char phase, const Event &event) {
			if (!first) {
				os << ",";
			}
			first = false;
			double timestamp_in_microseconds = event.timestamp_in_nanoseconds / 1000.0;
			os << "{\"name\":\"" << name << "\",\"ph\":\"" << phase
				<< "\",\"pid\":1,\"tid\":" << event.thread_id << ",\"ts\":" << timestamp_in_microseconds;
			if (event.phase == Phase::Counter) {
				os << ",\"args\":{\"value\":" << event.value << "}";
			}
			os << "}";
		};

		std::vector<const char *>names;
		{
			std::lock_guard<std::mutex> lock(g_names.mutex);
			names.reserve(g_names.names.size());
			for (auto &s : g_names.names) {
				names.push_back(s.c_str());
			}
		}

		for (auto &event : events) {
			const char *name = (event.id < names.size()) ? names[event.id] : "unknown";
			switch (event.phase) {
			case Phase::Begin: {
				emit(name, 'B', event);
			} break;
			case Phase::End: {
				emit(name, 'E', event);
			} break;
			case Phase::Counter: {
				emit(name, 'C', event);
			} break;
			case Phase::Mark: {
				emit(name, 'I', event);
			} break;
			}
		}
		os << "]}";
	}

	void flush_all() {
		std::vector<Event> all_events;
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			for (auto *recorder : g_all_recorders) {
				auto head = recorder->head.load(std::memory_order_relaxed);
				for (size_t i = recorder->tail; i < head; ++i) {
					all_events.push_back(recorder->event_buffer[i % recorder->capacity]);
				}
				recorder->tail = head;
			}
		}
		std::ofstream file(g_config.logfile_path, std::ios::out | std::ios::trunc);
		write_trace(file, all_events);
	}

	void shutdown() {
		flush_all();
	}
}
