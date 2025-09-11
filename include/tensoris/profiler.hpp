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
	bool enabled() noexcept;

	using NameId = uint32_t;

	NameId intern(std::string_view name) noexcept;
	uint64_t nanoseconds_now() noexcept;
	uint32_t time_id() noexcept; 

	void begin(NameId id) noexcept;
	void end(NameId id) noexcept;

	struct Scope {
		NameId id;
		uint64_t start_ns;
		explicit Scope(NameId id_) noexcept;
	};

	#define TPROF_SCOPE(name) ::tensoris_profile::Scope _tprof_scope_##__LINE__(::tensoris_profile::intern(name))
	#define TPROF_BEGIN(name) ::tensoris_profile::begin(::tensoris_profile::intern(name))
	#define TPROF_END(name) ::tensoris_profile::end(::tensoris_profile::intern(name))

	void counter(NameId id, int64_t value) noexcept;
	void mark(NameId id) noexcept;

	void flush_thread() noexcept;
	void flush_all();
}