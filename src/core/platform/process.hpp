#ifndef GUFO_CORE_PLATFORM_PROCESS_HPP_
#define GUFO_CORE_PLATFORM_PROCESS_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace gufo::platform {

// Paths preserve native Unicode arguments on Windows. Arguments exclude
// argv[0].
using ProcessArguments = std::vector<std::filesystem::path>;
using ProcessCancelled = std::function<bool()>;
using ProcessWrite = std::function<bool(std::span<const std::uint8_t>)>;
using ProcessProducer = std::function<void(const ProcessWrite&)>;
using ProcessInputArguments = std::function<ProcessArguments(
    const std::array<std::filesystem::path, 2>&)>;

struct ProcessInputResult {
  std::array<std::uint64_t, 2> bytes{};
  int status{-1};
  bool cancelled{false};
};

// Run a child with two concurrently produced input streams. The argument
// builder receives their input URLs (pipe descriptors or named pipe paths).
// Producers stop when write returns false. Cancellation is polled during I/O;
// an input failure stops the other producer and terminates the child.
// Returns false for spawn, I/O or wait errors; child exit and cancellation
// are reported in result. label identifies the program in diagnostics.
// Callbacks must not throw.
bool RunProcessWithInputs(const char* program, const char* label,
                          const ProcessInputArguments& arguments,
                          const std::array<ProcessProducer, 2>& producers,
                          const ProcessCancelled& cancelled,
                          ProcessInputResult* result, std::string* error);

// Capture at most 1 MiB of stdout, inheriting stderr. Waits for child exit.
bool CaptureProcessOutput(const char* program, ProcessArguments arguments,
                          std::string* output, std::string* error);

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PROCESS_HPP_
