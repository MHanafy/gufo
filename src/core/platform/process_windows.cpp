#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/platform/process.hpp"

namespace gufo::platform {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;
constexpr DWORD kPollMilliseconds = 50;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string WindowsMessage(DWORD code) {
  return std::system_category().message(static_cast<int>(code));
}

std::wstring Widen(std::string_view text) {
  if (text.empty()) {
    return {};
  }
  const int length = MultiByteToWideChar(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(std::max(length, 0)), L'\0');
  if (length > 0) {
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                        wide.data(), length);
  }
  return wide;
}

class Handle {
public:
  Handle() = default;
  explicit Handle(HANDLE handle)
      : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
  ~Handle() { Reset(); }
  Handle(Handle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  [[nodiscard]] HANDLE get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }
  void Reset() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
      handle_ = nullptr;
    }
  }

private:
  HANDLE handle_{nullptr};
};

std::wstring PipeName(std::wstring_view stream) {
  static std::atomic<std::uint64_t> sequence{0};
  return L"\\\\.\\pipe\\gufo-h3-" + std::to_wstring(GetCurrentProcessId()) +
         L"-" +
         std::to_wstring(sequence.fetch_add(1, std::memory_order_relaxed)) +
         L"-" + std::wstring(stream);
}

bool IsCancelled(const ProcessCancelled* cancellation) {
  return cancellation != nullptr && *cancellation && (*cancellation)();
}

// CommandLineToArgvW quoting, so FFmpeg sees each argument unchanged.
void AppendArgument(std::wstring* line, std::wstring_view argument) {
  if (!line->empty()) {
    line->push_back(L' ');
  }
  if (!argument.empty() &&
      argument.find_first_of(L" \t\n\v\"") == std::wstring_view::npos) {
    line->append(argument);
    return;
  }
  line->push_back(L'"');
  for (auto it = argument.begin();; ++it) {
    std::size_t backslashes = 0;
    while (it != argument.end() && *it == L'\\') {
      ++it;
      ++backslashes;
    }
    if (it == argument.end()) {
      line->append(backslashes * 2, L'\\');
      break;
    }
    if (*it == L'"') {
      line->append(backslashes * 2 + 1, L'\\');
    } else {
      line->append(backslashes, L'\\');
    }
    line->push_back(*it);
  }
  line->push_back(L'"');
}

Handle InheritableNul(DWORD access) {
  SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
  return Handle(CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &attributes, OPEN_EXISTING, 0, nullptr));
}

// The parent's standard handle as an inheritable copy, or NUL without one.
Handle InheritableStandard(DWORD which) {
  const HANDLE source = GetStdHandle(which);
  HANDLE copy = nullptr;
  if (source != nullptr && source != INVALID_HANDLE_VALUE &&
      DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &copy,
                      0, TRUE, DUPLICATE_SAME_ACCESS)) {
    return Handle(copy);
  }
  return InheritableNul(GENERIC_WRITE);
}

// Starts `program` with `arguments` (argv[1..]); errors name it `label`. stdin
// is NUL; stdout is `output` when given, else the parent's. Only these three
// handles are inherited.
bool Spawn(const char* program, std::string_view label,
           const std::vector<std::wstring>& arguments, HANDLE output,
           Handle* process, std::string* error) {
  std::wstring line;
  AppendArgument(&line, Widen(program));
  for (const std::wstring& argument : arguments) {
    AppendArgument(&line, argument);
  }
  Handle input = InheritableNul(GENERIC_READ);
  Handle parent_output;
  if (output == nullptr) {
    parent_output = InheritableStandard(STD_OUTPUT_HANDLE);
    output = parent_output.get();
  }
  Handle errors = InheritableStandard(STD_ERROR_HANDLE);
  if (!input || output == nullptr || !errors) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::array<HANDLE, 3> inherited = {input.get(), output, errors.get()};
  SIZE_T attribute_bytes = 0;
  (void)InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
  std::vector<unsigned char> attribute_storage(attribute_bytes);
  auto* attributes =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage.data());
  if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attribute_bytes)) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = output;
  startup.StartupInfo.hStdError = errors.get();
  startup.lpAttributeList = attributes;
  PROCESS_INFORMATION info{};
  const bool created =
      UpdateProcThreadAttribute(
          attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited.data(),
          sizeof(inherited), nullptr, nullptr) &&
      CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                     EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                     nullptr, &startup.StartupInfo, &info);
  const DWORD failure = created ? ERROR_SUCCESS : GetLastError();
  DeleteProcThreadAttributeList(attributes);
  if (!created) {
    SetError(error, "cannot start " + std::string(label) + ": " +
                        WindowsMessage(failure));
    return false;
  }
  CloseHandle(info.hThread);
  *process = Handle(info.hProcess);
  return true;
}

bool WaitForChild(HANDLE child, int* status, std::string* error) {
  DWORD code = 0;
  if (WaitForSingleObject(child, INFINITE) != WAIT_OBJECT_0 ||
      !GetExitCodeProcess(child, &code)) {
    SetError(error, "cannot wait for media process: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  *status = static_cast<int>(code);
  return true;
}

Handle CreateOutboundPipe(const std::wstring& name) {
  return Handle(
      CreateNamedPipeW(name.c_str(),
                       PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED |
                           FILE_FLAG_FIRST_PIPE_INSTANCE,
                       PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                       1, static_cast<DWORD>(kIoChunkBytes), 0, 0, nullptr));
}

// Completes an overlapped pipe operation, or cancels it when the stream is
// stopped, the job is cancelled or FFmpeg has exited.
bool AwaitOverlapped(HANDLE pipe, OVERLAPPED* overlapped, HANDLE child,
                     const ProcessCancelled* cancellation,
                     const std::atomic<bool>& stop, DWORD* transferred,
                     DWORD* failure) {
  const std::array<HANDLE, 2> waits = {overlapped->hEvent, child};
  while (true) {
    const DWORD result =
        WaitForMultipleObjects(static_cast<DWORD>(waits.size()), waits.data(),
                               FALSE, kPollMilliseconds);
    if (result == WAIT_OBJECT_0) {
      if (GetOverlappedResult(pipe, overlapped, transferred, FALSE)) {
        return true;
      }
      *failure = GetLastError();
      return false;
    }
    const DWORD wait_failure =
        result == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
    const bool child_exited = result == WAIT_OBJECT_0 + 1;
    if (child_exited || result == WAIT_FAILED ||
        stop.load(std::memory_order_acquire) || IsCancelled(cancellation)) {
      CancelIoEx(pipe, overlapped);
      if (GetOverlappedResult(pipe, overlapped, transferred, TRUE)) {
        return true;
      }
      *failure = child_exited            ? ERROR_BROKEN_PIPE
                 : result == WAIT_FAILED ? wait_failure
                                         : ERROR_OPERATION_ABORTED;
      return false;
    }
  }
}

bool ConnectPipe(HANDLE pipe, HANDLE child,
                 const ProcessCancelled* cancellation,
                 const std::atomic<bool>& stop, DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  OVERLAPPED overlapped{};
  overlapped.hEvent = event.get();
  if (ConnectNamedPipe(pipe, &overlapped)) {
    return true;
  }
  const DWORD code = GetLastError();
  if (code == ERROR_PIPE_CONNECTED) {
    return true;
  }
  if (code != ERROR_IO_PENDING) {
    *failure = code;
    return false;
  }
  DWORD ignored = 0;
  return AwaitOverlapped(pipe, &overlapped, child, cancellation, stop, &ignored,
                         failure);
}

bool WriteAll(HANDLE pipe, const std::uint8_t* data, std::size_t bytes,
              HANDLE child, const ProcessCancelled* cancellation,
              const std::atomic<bool>& stop, std::uint64_t* written,
              DWORD* failure) {
  Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!event) {
    *failure = GetLastError();
    return false;
  }
  while (bytes != 0) {
    if (stop.load(std::memory_order_acquire) || IsCancelled(cancellation)) {
      *failure = ERROR_OPERATION_ABORTED;
      return false;
    }
    OVERLAPPED overlapped{};
    overlapped.hEvent = event.get();
    ResetEvent(event.get());
    const auto request =
        static_cast<DWORD>(std::min<std::size_t>(bytes, kIoChunkBytes));
    if (!WriteFile(pipe, data, request, nullptr, &overlapped)) {
      const DWORD code = GetLastError();
      if (code != ERROR_IO_PENDING) {
        *failure = code;
        return false;
      }
    }
    DWORD amount = 0;
    if (!AwaitOverlapped(pipe, &overlapped, child, cancellation, stop, &amount,
                         failure)) {
      return false;
    }
    if (amount == 0) {
      *failure = ERROR_BROKEN_PIPE;
      return false;
    }
    data += amount;
    bytes -= amount;
    *written += amount;
  }
  return true;
}

}  // namespace

bool CaptureProcessOutput(const char* program, ProcessArguments paths,
                          std::string* output, std::string* error) {
  std::vector<std::wstring> arguments;
  for (const auto& path : paths) {
    arguments.push_back(path.wstring());
  }
  SECURITY_ATTRIBUTES inheritable{sizeof(inheritable), nullptr, TRUE};
  HANDLE read_end = nullptr;
  HANDLE write_end = nullptr;
  if (!CreatePipe(&read_end, &write_end, &inheritable, 0)) {
    SetError(error, "cannot create media inspection pipe: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  Handle reader(read_end);
  Handle writer(write_end);
  SetHandleInformation(reader.get(), HANDLE_FLAG_INHERIT, 0);
  Handle child;
  const bool started =
      Spawn(program, program, arguments, writer.get(), &child, error);
  writer.Reset();
  if (!started) {
    return false;
  }
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    DWORD amount = 0;
    if (!ReadFile(reader.get(), buffer.data(),
                  static_cast<DWORD>(buffer.size()), &amount, nullptr) ||
        amount == 0) {
      break;
    }
    if (output->size() + amount > (1U << 20U)) {
      reader.Reset();
      TerminateProcess(child.get(), 1);
      int ignored = 0;
      (void)WaitForChild(child.get(), &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), amount);
  }
  reader.Reset();
  int status = 0;
  if (!WaitForChild(child.get(), &status, error)) {
    return false;
  }
  if (status != 0) {
    SetError(error, std::string(program) + " exited with status " +
                        std::to_string(status));
    return false;
  }
  return true;
}

bool RunProcessWithInputs(const char* program, const char* label,
                          const ProcessInputArguments& arguments,
                          const std::array<ProcessProducer, 2>& producers,
                          const ProcessCancelled& cancelled,
                          ProcessInputResult* result, std::string* error) {
  *result = {};
  const std::wstring video_name = PipeName(L"input0");
  const std::wstring audio_name = PipeName(L"input1");
  std::array<Handle, 2> pipes{CreateOutboundPipe(video_name),
                              CreateOutboundPipe(audio_name)};
  if (!pipes[0] || !pipes[1]) {
    SetError(error, "cannot create process input pipes: " +
                        WindowsMessage(GetLastError()));
    return false;
  }
  std::vector<std::wstring> native_arguments;
  for (const auto& path : arguments({video_name, audio_name})) {
    native_arguments.push_back(path.wstring());
  }
  Handle child;
  if (!Spawn(program, label, native_arguments, nullptr, &child, error)) {
    return false;
  }
  std::atomic<bool> stop{false};
  std::array<DWORD, 2> failures{};
  std::array<std::thread, 2> threads;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      DWORD failure = ERROR_SUCCESS;
      const auto record = [&] {
        if (!IsCancelled(&cancelled) && !stop.load(std::memory_order_acquire)) {
          failures[i] = failure;
        }
        stop.store(true, std::memory_order_release);
      };
      if (!ConnectPipe(pipes[i].get(), child.get(), &cancelled, stop,
                       &failure)) {
        record();
      } else {
        producers[i]([&](std::span<const std::uint8_t> bytes) {
          if (!WriteAll(pipes[i].get(), bytes.data(), bytes.size(), child.get(),
                        &cancelled, stop, &result->bytes[i], &failure)) {
            record();
            return false;
          }
          return true;
        });
      }
      pipes[i].Reset();
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  result->cancelled = IsCancelled(&cancelled);
  if (stop.load(std::memory_order_acquire) || result->cancelled) {
    TerminateProcess(child.get(), 1);
  }
  if (!WaitForChild(child.get(), &result->status, error)) {
    return false;
  }
  const DWORD writer_error =
      failures[0] != ERROR_SUCCESS ? failures[0] : failures[1];
  if (!result->cancelled && writer_error != ERROR_SUCCESS) {
    SetError(error, "cannot stream generated media to " + std::string(label) +
                        ": " + WindowsMessage(writer_error));
    return false;
  }
  return true;
}

}  // namespace gufo::platform
