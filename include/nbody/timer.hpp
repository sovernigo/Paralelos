#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace nbody {

// Cronometro monotonico (steady_clock): mede tempo de parede.
class Timer {
 public:
  void start() { start_ = std::chrono::steady_clock::now(); }

  [[nodiscard]] double stop_ms() const {
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_{};
};

struct TimingStats {
  double best_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
};

// Melhor amostra e mediana sao mais estaveis que a media em maquinas com
// ruido de agendamento.
inline TimingStats summarize(std::vector<double> samples) {
  TimingStats stats;
  if (samples.empty()) {
    return stats;
  }
  stats.best_ms = *std::min_element(samples.begin(), samples.end());
  stats.mean_ms =
      std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
  std::sort(samples.begin(), samples.end());
  const std::size_t mid = samples.size() / 2;
  stats.median_ms =
      (samples.size() % 2 == 0) ? 0.5 * (samples[mid - 1] + samples[mid]) : samples[mid];
  return stats;
}

}  // namespace nbody
