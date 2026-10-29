#include "src/models/minimax_h3/media.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/platform/os_info.hpp"
#include "src/core/platform/process.hpp"

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

namespace gufo::minimax_h3 {
namespace {

constexpr int kAudioChunkSamples = 4096;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

const char* FfmpegProgram() {
  const char* override = std::getenv("GUFO_FFMPEG");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFMPEG_EXECUTABLE;
}

const char* FfprobeProgram() {
  const char* override = std::getenv("GUFO_FFPROBE");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFPROBE_EXECUTABLE;
}

std::filesystem::path PartialMediaPath(
    const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> sequence{0};
  std::filesystem::path name = destination.filename();
  name += ".gufo-partial-" + std::to_string(platform::ProcessId()) + "-" +
          std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) +
          ".mp4";
  return destination.parent_path() / name;
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

bool ParseInteger(std::string_view text, int* output) {
  if (text.empty()) {
    return false;
  }
  int value = 0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseDouble(std::string_view text, double* output) {
  if (text.empty() || text == "N/A") {
    *output = 0.0;
    return true;
  }
  double value = 0.0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(value)) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseRate(std::string_view text, double* output) {
  const std::size_t slash = text.find('/');
  if (slash == std::string_view::npos) {
    return ParseDouble(text, output);
  }
  double numerator = 0.0;
  double denominator = 0.0;
  if (!ParseDouble(text.substr(0, slash), &numerator) ||
      !ParseDouble(text.substr(slash + 1), &denominator) ||
      denominator == 0.0) {
    return false;
  }
  *output = numerator / denominator;
  return std::isfinite(*output);
}

using KeyValues = std::vector<std::pair<std::string, std::string>>;

KeyValues ParseKeyValues(std::string_view text) {
  KeyValues result;
  while (!text.empty()) {
    const std::size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const std::size_t equals = line.find('=');
    if (equals != std::string_view::npos) {
      result.emplace_back(line.substr(0, equals), line.substr(equals + 1));
    }
    if (newline == std::string_view::npos) {
      break;
    }
    text.remove_prefix(newline + 1);
  }
  return result;
}

const std::string* FindValue(const KeyValues& values, std::string_view key) {
  for (const auto& [candidate, value] : values) {
    if (candidate == key) {
      return &value;
    }
  }
  return nullptr;
}

}  // namespace

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          const MediaVideoOptions& video,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (path.empty() || frame_count < 1 || video.input_width < 2 ||
      video.input_height < 2 || video.output_width < 2 ||
      video.output_height < 2 || (video.input_width & 1) != 0 ||
      (video.input_height & 1) != 0 || (video.output_width & 1) != 0 ||
      (video.output_height & 1) != 0 || video.fps < 1 || samples < 1 ||
      channels < 1 || sample_rate < 1) {
    SetError(error, "invalid MiniMax H3 media mux arguments");
    return false;
  }
  const std::uint64_t frame_bytes =
      static_cast<std::uint64_t>(video.input_width) * video.input_height * 3U;
  const std::uint64_t video_bytes =
      frame_bytes * static_cast<std::uint64_t>(frame_count);
  const std::uint64_t audio_elements =
      static_cast<std::uint64_t>(samples) * channels;
  if (video_bytes != rgb24.size() ||
      audio_elements != channel_major_pcm.size() ||
      video_bytes > std::numeric_limits<std::size_t>::max() ||
      audio_elements >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    SetError(error, "MiniMax H3 media stream geometry differs from buffers");
    return false;
  }
  if (IsCancelled(cancellation)) {
    if (telemetry != nullptr) {
      telemetry->cancelled = true;
    }
    SetError(error, "MiniMax H3 media mux cancelled before start");
    return false;
  }
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create media output directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial_path = PartialMediaPath(path);
  std::filesystem::remove(partial_path, filesystem_error);
  filesystem_error.clear();

  const std::string size = std::to_string(video.input_width) + "x" +
                           std::to_string(video.input_height);
  const std::string output_size = std::to_string(video.output_width) + ":" +
                                  std::to_string(video.output_height);
  const std::string rate = std::to_string(video.fps);
  const std::string audio_rate = std::to_string(sample_rate);
  const std::string audio_channels = std::to_string(channels);
  const auto arguments =
      [&](const std::array<std::filesystem::path, 2>& inputs) {
        return platform::ProcessArguments{
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            size,
            "-framerate",
            rate,
            "-i",
            inputs[0],
            "-f",
            "f32le",
            "-ar",
            audio_rate,
            "-ac",
            audio_channels,
            "-i",
            inputs[1],
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            "scale=" + output_size + ":flags=lanczos,setsar=1",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            partial_path,
        };
      };
  std::size_t maximum_audio_staging = 0;
  const std::array<platform::ProcessProducer, 2> producers = {
      [&](const platform::ProcessWrite& write) { (void)write(rgb24); },
      [&](const platform::ProcessWrite& write) {
        std::vector<float> interleaved(
            static_cast<std::size_t>(kAudioChunkSamples) * channels);
        maximum_audio_staging = interleaved.size() * sizeof(float);
        int offset = 0;
        while (offset < samples) {
          const int chunk = std::min(kAudioChunkSamples, samples - offset);
          for (int sample = 0; sample < chunk; ++sample) {
            for (int channel = 0; channel < channels; ++channel) {
              interleaved[static_cast<std::size_t>(sample) * channels +
                          channel] =
                  channel_major_pcm[static_cast<std::size_t>(channel) *
                                        samples +
                                    offset + sample];
            }
          }
          if (!write({reinterpret_cast<const std::uint8_t*>(interleaved.data()),
                      static_cast<std::size_t>(chunk) * channels *
                          sizeof(float)})) {
            break;
          }
          offset += chunk;
        }
      },
  };
  platform::ProcessInputResult result;
  const bool streamed = platform::RunProcessWithInputs(
      FfmpegProgram(), "FFmpeg", arguments, producers,
      [&] { return IsCancelled(cancellation); }, &result, error);
  const auto written_video = result.bytes[0];
  const auto written_audio = result.bytes[1];
  if (telemetry != nullptr) {
    telemetry->video_bytes = written_video;
    telemetry->audio_bytes = written_audio;
    telemetry->maximum_audio_staging_bytes = maximum_audio_staging;
    telemetry->child_status = result.status;
    telemetry->cancelled = result.cancelled;
  }
  if (!streamed || result.cancelled || result.status != 0) {
    std::filesystem::remove(partial_path, filesystem_error);
    if (result.cancelled) {
      SetError(error, "MiniMax H3 media mux cancelled");
    } else if (streamed) {
      SetError(error,
               "FFmpeg exited with status " + std::to_string(result.status));
    }
    return false;
  }
  if (written_video != video_bytes ||
      written_audio != audio_elements * sizeof(float)) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "FFmpeg media streams ended before all bytes were sent");
    return false;
  }
  MediaProbe probe;
  const double expected_video_duration =
      static_cast<double>(frame_count) / video.fps;
  const double expected_audio_duration =
      static_cast<double>(samples) / sample_rate;
  if (!ProbeMediaFile(partial_path, &probe, error) ||
      probe.video_codec != "h264" || probe.audio_codec != "aac" ||
      probe.width != video.output_width ||
      probe.height != video.output_height ||
      std::abs(probe.video_frame_rate - video.fps) > 1.0e-6 ||
      probe.audio_channels != channels ||
      probe.audio_sample_rate != sample_rate ||
      std::abs(probe.video_start_seconds - probe.audio_start_seconds) >
          1.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - expected_video_duration) >
          1.0 / video.fps ||
      std::abs(probe.audio_duration_seconds - expected_audio_duration) >
          2048.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - probe.audio_duration_seconds) >
          0.075) {
    std::filesystem::remove(partial_path, filesystem_error);
    if (error != nullptr && error->empty()) {
      *error = "generated MiniMax H3 media violates the delivery contract";
    }
    return false;
  }
  std::filesystem::rename(partial_path, path, filesystem_error);
  if (filesystem_error) {
    const std::string message = filesystem_error.message();
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot publish generated media: " + message);
    return false;
  }
  return true;
}

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          int width, int height, int fps,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  return WriteSynchronizedMp4(path, rgb24, frame_count,
                              {.input_width = width,
                               .input_height = height,
                               .output_width = width,
                               .output_height = height,
                               .fps = fps},
                              channel_major_pcm, samples, channels, sample_rate,
                              cancellation, telemetry, error);
}

bool ProbeMediaFile(const std::filesystem::path& path, MediaProbe* probe,
                    std::string* error) {
  if (path.empty() || probe == nullptr) {
    SetError(error, "invalid MiniMax H3 media probe arguments");
    return false;
  }
  *probe = {};
  std::string video_output;
  platform::ProcessArguments video_arguments = {
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=codec_name,width,height,r_frame_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path,
  };
  if (!platform::CaptureProcessOutput(
          FfprobeProgram(), std::move(video_arguments), &video_output, error)) {
    return false;
  }
  const KeyValues video = ParseKeyValues(video_output);
  const std::string* video_codec = FindValue(video, "codec_name");
  const std::string* width = FindValue(video, "width");
  const std::string* height = FindValue(video, "height");
  const std::string* frame_rate = FindValue(video, "r_frame_rate");
  const std::string* video_start = FindValue(video, "start_time");
  const std::string* video_duration = FindValue(video, "duration");
  if (video_codec == nullptr || width == nullptr || height == nullptr ||
      frame_rate == nullptr || video_start == nullptr ||
      video_duration == nullptr || !ParseInteger(*width, &probe->width) ||
      !ParseInteger(*height, &probe->height) ||
      !ParseRate(*frame_rate, &probe->video_frame_rate) ||
      !ParseDouble(*video_start, &probe->video_start_seconds) ||
      !ParseDouble(*video_duration, &probe->video_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 video metadata");
    return false;
  }
  probe->video_codec = *video_codec;

  std::string audio_output;
  platform::ProcessArguments audio_arguments = {
      "-v",
      "error",
      "-select_streams",
      "a:0",
      "-show_entries",
      "stream=codec_name,channels,sample_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path,
  };
  if (!platform::CaptureProcessOutput(
          FfprobeProgram(), std::move(audio_arguments), &audio_output, error)) {
    return false;
  }
  const KeyValues audio = ParseKeyValues(audio_output);
  const std::string* audio_codec = FindValue(audio, "codec_name");
  const std::string* audio_channels = FindValue(audio, "channels");
  const std::string* audio_rate = FindValue(audio, "sample_rate");
  const std::string* audio_start = FindValue(audio, "start_time");
  const std::string* audio_duration = FindValue(audio, "duration");
  if (audio_codec == nullptr || audio_channels == nullptr ||
      audio_rate == nullptr || audio_start == nullptr ||
      audio_duration == nullptr ||
      !ParseInteger(*audio_channels, &probe->audio_channels) ||
      !ParseInteger(*audio_rate, &probe->audio_sample_rate) ||
      !ParseDouble(*audio_start, &probe->audio_start_seconds) ||
      !ParseDouble(*audio_duration, &probe->audio_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 audio metadata");
    return false;
  }
  probe->audio_codec = *audio_codec;
  return true;
}

}  // namespace gufo::minimax_h3
