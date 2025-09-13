#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <string_view>

namespace tensoris_profile {
	struct Config {
		bool enabled = false;
		size_t per_thread_capacity = 1 << 15;
		const char *logfile_path = "tensoris_trace.json";
		bool enable_os_signpost = false;
	};

	void init_config(const Config &config);
	void shutdown();
	void enable(bool on);
	bool is_enabled() noexcept;

	using NameId = uint32_t;

	NameId intern(std::string_view name) noexcept;
	uint64_t now_in_nanoseconds() noexcept;
	uint32_t thread_id() noexcept; 

	void begin_scope(NameId id) noexcept;
	void end_scope(NameId id) noexcept;

	struct Scope {
		NameId scope_name;
		uint64_t start_ns;
		explicit Scope(NameId scope_name) noexcept;
		~Scope() noexcept;
	};

	#define TPROF_SCOPE(name) ::tensoris_profile::Scope _tprof_scope_##__LINE__(::tensoris_profile::intern(name))
	#define TPROF_BEGIN(name) ::tensoris_profile::begin(::tensoris_profile::intern(name))
	#define TPROF_END(name) ::tensoris_profile::end(::tensoris_profile::intern(name))

	void event_counter(NameId id, int64_t value) noexcept;
	void event_mark(NameId id) noexcept;

	void flush_thread() noexcept;
	void flush_all();
}