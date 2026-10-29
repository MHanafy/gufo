#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/platform/process.hpp"

extern char** environ;

namespace gufo::platform {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;

bool EnsureSigpipeIgnored(std::string* error) {
  static std::once_flag once;
  static int failure = 0;
  std::call_once(once, [] {
    struct sigaction ignore{};
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    if (sigaction(SIGPIPE, &ignore, nullptr) != 0) {
      failure = errno;
    }
  });
  if (failure == 0) {
    return true;
  }
  if (error != nullptr) {
    *error = "cannot ignore SIGPIPE for FFmpeg writers: " +
             std::string(std::strerror(failure));
  }
  return false;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const ProcessCancelled* cancellation) {
  return cancellation != nullptr && *cancellation && (*cancellation)();
}

void CloseFd(int* descriptor) {
  if (*descriptor >= 0) {
    while (close(*descriptor) != 0 && errno == EINTR) {
    }
    *descriptor = -1;
  }
}

bool MakeNonBlocking(int descriptor, std::string* error) {
  const int flags = fcntl(descriptor, F_GETFL);
  if (flags < 0 || fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0) {
    SetError(error, "cannot configure FFmpeg media pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

ssize_t CancellableWrite(int descriptor, const void* data, std::size_t bytes,
                         const ProcessCancelled* cancellation,
                         const std::atomic<bool>& stop) {
  while (!stop.load(std::memory_order_acquire) && !IsCancelled(cancellation)) {
    const ssize_t amount = write(descriptor, data, bytes);
    if (amount >= 0) {
      return amount;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return -1;
    }
    struct pollfd readiness{descriptor, POLLOUT, 0};
    const int result = poll(&readiness, 1, 50);
    if (result < 0 && errno != EINTR) {
      return -1;
    }
    if (result > 0 && (readiness.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      errno = EPIPE;
      return -1;
    }
  }
  errno = ECANCELED;
  return -1;
}

int AddCloseUnlessKept(posix_spawn_file_actions_t* actions, int descriptor,
                       int keep_a, int keep_b) {
  if (descriptor == keep_a || descriptor == keep_b) {
    return 0;
  }
  return posix_spawn_file_actions_addclose(actions, descriptor);
}

bool WaitForChild(pid_t child, int* status, std::string* error) {
  while (waitpid(child, status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    SetError(error, "cannot wait for media process: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

std::vector<char*> MutableArgv(std::span<std::string> storage) {
  std::vector<char*> arguments;
  arguments.reserve(storage.size() + 1);
  for (std::string& argument : storage) {
    arguments.push_back(argument.data());
  }
  arguments.push_back(nullptr);
  return arguments;
}

}  // namespace

bool CaptureProcessOutput(const char* program, ProcessArguments paths,
                          std::string* output, std::string* error) {
  std::vector<std::string> storage{program};
  for (const auto& path : paths) {
    storage.push_back(path.string());
  }
  std::vector<char*> arguments = MutableArgv(storage);
  int stream[2] = {-1, -1};
  if (pipe(stream) != 0) {
    SetError(error, "cannot create media inspection pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code = posix_spawn_file_actions_adddup2(&actions, stream[1], STDOUT_FILENO);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[0]);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[1]);
  }
  pid_t child = -1;
  if (code == 0) {
    code = posix_spawnp(&child, program, &actions, nullptr, arguments.data(),
                        environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&stream[1]);
  if (code != 0) {
    CloseFd(&stream[0]);
    SetError(error, "cannot start " + std::string(program) + ": " +
                        std::string(std::strerror(code)));
    return false;
  }
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    const ssize_t amount = read(stream[0], buffer.data(), buffer.size());
    if (amount < 0 && errno == EINTR) {
      continue;
    }
    if (amount <= 0) {
      break;
    }
    if (output->size() + static_cast<std::size_t>(amount) > (1U << 20U)) {
      CloseFd(&stream[0]);
      (void)kill(child, SIGTERM);
      int ignored = 0;
      (void)WaitForChild(child, &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), static_cast<std::size_t>(amount));
  }
  CloseFd(&stream[0]);
  int status = 0;
  if (!WaitForChild(child, &status, error)) {
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    SetError(error,
             std::string(program) + " exited with status " +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
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
  int video_pipe[2] = {-1, -1};
  int audio_pipe[2] = {-1, -1};
  if (pipe(video_pipe) != 0 || pipe(audio_pipe) != 0) {
    const int saved = errno;
    CloseFd(&video_pipe[0]);
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[0]);
    CloseFd(&audio_pipe[1]);
    SetError(error, "cannot create process input pipes: " +
                        std::string(std::strerror(saved)));
    return false;
  }
  const int audio_target =
      std::max({video_pipe[0], video_pipe[1], audio_pipe[0], audio_pipe[1]}) +
      1;
  const auto paths =
      arguments({"pipe:0", "pipe:" + std::to_string(audio_target)});
  std::vector<std::string> storage{program};
  for (const auto& path : paths) {
    storage.push_back(path.string());
  }
  std::vector<char*> argv = MutableArgv(storage);
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, video_pipe[0], STDIN_FILENO);
  }
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, audio_pipe[0], audio_target);
  }
  const std::array<int, 4> descriptors = {video_pipe[0], video_pipe[1],
                                          audio_pipe[0], audio_pipe[1]};
  for (const int descriptor : descriptors) {
    if (code == 0) {
      code =
          AddCloseUnlessKept(&actions, descriptor, STDIN_FILENO, audio_target);
    }
  }
  pid_t child = -1;
  if (code == 0) {
    code =
        posix_spawnp(&child, program, &actions, nullptr, argv.data(), environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&video_pipe[0]);
  CloseFd(&audio_pipe[0]);
  if (code != 0) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    SetError(error, "cannot start " + std::string(label) + ": " +
                        std::string(std::strerror(code)));
    return false;
  }
  if (!MakeNonBlocking(video_pipe[1], error) ||
      !MakeNonBlocking(audio_pipe[1], error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    return false;
  }

  std::atomic<bool> stop{false};
  std::array<int, 2> failures{};
  if (!EnsureSigpipeIgnored(error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    return false;
  }
  std::array<int, 2> writers{video_pipe[1], audio_pipe[1]};
  std::array<std::thread, 2> threads;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      producers[i]([&](std::span<const std::uint8_t> bytes) {
        const std::uint8_t* cursor = bytes.data();
        std::size_t remaining = bytes.size();
        while (remaining != 0 && !stop.load(std::memory_order_acquire)) {
          if (IsCancelled(&cancelled)) {
            stop.store(true, std::memory_order_release);
            break;
          }
          const std::size_t request = std::min(remaining, kIoChunkBytes);
          const ssize_t amount =
              CancellableWrite(writers[i], cursor, request, &cancelled, stop);
          if (amount <= 0) {
            if (!IsCancelled(&cancelled) &&
                !stop.load(std::memory_order_acquire)) {
              failures[i] = amount < 0 ? errno : EIO;
            }
            stop.store(true, std::memory_order_release);
            break;
          }
          cursor += amount;
          remaining -= static_cast<std::size_t>(amount);
          result->bytes[i] += static_cast<std::uint64_t>(amount);
        }
        return remaining == 0 && !stop.load(std::memory_order_acquire);
      });
      CloseFd(&writers[i]);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  result->cancelled = IsCancelled(&cancelled);
  if (stop.load(std::memory_order_acquire) || result->cancelled) {
    (void)kill(child, SIGTERM);
  }
  int status = 0;
  if (!WaitForChild(child, &status, error)) {
    return false;
  }
  result->status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  const int writer_error = failures[0] != 0 ? failures[0] : failures[1];
  if (!result->cancelled && writer_error != 0) {
    SetError(error, "cannot stream generated media to " + std::string(label) +
                        ": " + std::string(std::strerror(writer_error)));
    return false;
  }
  return true;
}

}  // namespace gufo::platform
