#ifndef CYPERSTEREO_THREAD_PRIORITY_H_
#define CYPERSTEREO_THREAD_PRIORITY_H_

// ============================================================================
//  Pipeline thread scheduling  ——  单一调整入口 (edit values in one place)
// ============================================================================
// All pipeline threads use the real-time SCHED_FIFO policy on Linux. This works
// identically on x86 and on every ARM target (RK3588 / Horizon Sunrise X5 /
// Raspberry Pi, ...) because it only relies on POSIX pthreads.
//
// SCHED_FIFO priorities range 1 (low) .. 99 (high). Ordinary (non-RT) system
// threads run below all of these, so every thread configured here sits well
// above the rest of the system.
//
// Ordering rationale (higher value = preempts lower):
//   poll   (USB capture)  : HIGHEST. A late VIDIOC_DQBUF = a permanently
//                           dropped frame, so capture must be able to preempt
//                           everything, including the workers.
//   worker (4x wb+cvt)    : just below poll — runs promptly but can never
//                           starve capture.
//   main   (orchestration): just below the workers (printing / imshow /
//                           bookkeeping); least time-critical.
//
// NOTE on "poll and workers both highest": equal-priority SCHED_FIFO threads do
// NOT preempt one another. If the 4 workers were at the SAME priority as poll
// and happened to saturate all cores, a URB arriving mid-burst could wait up to
// a full worker runtime before poll got a core -> a dropped frame. Keeping poll
// one notch above the workers removes that risk while still putting capture +
// processing far above every other thread. If you truly want them equal, set
// kPollPriority == kWorkerPriority below.

#include <iostream>
#include <mutex>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <cstring>
#else
#ifndef SCHED_FIFO
#define SCHED_FIFO 0
#endif
#ifndef SCHED_RR
#define SCHED_RR 0
#endif
#endif

#include "cyperstereo.h"

CYPERSTEREO_BEGIN_NAMESPACE

enum class ThreadRole { kPoll, kWorker, kMain };

struct ThreadPriorityConfig {
  bool enable          = true;        // false -> keep default OS scheduling
  int  policy          = SCHED_FIFO;  // SCHED_FIFO (recommended) or SCHED_RR
  int  poll_priority   = 82;          // USB capture thread   (highest)
  int  worker_priority = 80;          // camera wb+cvt workers (just below poll)
  int  main_priority   = 78;          // main / orchestration  (just below workers)
};

// Single global config instance. Tweak the fields here to retune the whole
// pipeline, or mutate GetThreadPriorityConfig() at startup before threads run.
inline ThreadPriorityConfig& GetThreadPriorityConfig() {
  static ThreadPriorityConfig cfg;
  return cfg;
}

inline int PriorityForRole(ThreadRole role) {
  const ThreadPriorityConfig& c = GetThreadPriorityConfig();
  switch (role) {
    case ThreadRole::kPoll:   return c.poll_priority;
    case ThreadRole::kWorker: return c.worker_priority;
    case ThreadRole::kMain:   return c.main_priority;
  }
  return c.main_priority;
}

// Pin the CURRENT thread to one logical CPU. x86 gives each of the 4 ISP
// workers its own physical core (CPU0-3, with SMT siblings CPU4-7). A board
// that declares a known topology may also use it: TARGET_BOARD=rk3588 pins
// the workers to its Cortex-A76 cluster CPU4-7. Unknown ARM big.LITTLE
// layouts remain unpinned rather than guessing and landing on LITTLE cores.
inline void PinThreadToCpu(int cpu) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu;
#endif
}

// Restrict the CURRENT thread to an inclusive CPU range. Threads created
// afterwards inherit this mask on Linux, so applications can establish a
// process-wide cluster boundary before starting worker or capture threads.
// Returns a pthread error number (zero on success).
inline int RestrictCurrentThreadToCpuRange(int first_cpu, int last_cpu) {
#if defined(__linux__)
  if (first_cpu < 0 || last_cpu < first_cpu || last_cpu >= CPU_SETSIZE)
    return EINVAL;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu = first_cpu; cpu <= last_cpu; ++cpu)
    CPU_SET(cpu, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0)
    return rc;

  // Linux may intersect the requested mask with a cpuset/cgroup constraint.
  // Treat a silently narrowed mask as failure: the caller asked for this
  // exact cluster, not merely any non-empty subset of it.
  cpu_set_t actual;
  CPU_ZERO(&actual);
  rc = pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual);
  if (rc != 0)
    return rc;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    const bool expected = cpu >= first_cpu && cpu <= last_cpu;
    if ((CPU_ISSET(cpu, &actual) != 0) != expected)
      return EINVAL;
  }
  return 0;
#else
  (void)first_cpu;
  (void)last_cpu;
  return 0;
#endif
}

// Temporarily bind the CURRENT thread to one CPU, then restore the affinity
// mask it had on entry. This is used when the caller itself executes one shard
// of a multi-camera batch: worker threads can stay permanently pinned, while a
// caller owned by an application must not retain an SDK-specific affinity.
class ScopedThreadAffinity {
 public:
  explicit ScopedThreadAffinity(int cpu) {
#if defined(__linux__)
    if (cpu < 0 ||
        pthread_getaffinity_np(pthread_self(), sizeof(saved_), &saved_) != 0)
      return;
    cpu_set_t target;
    CPU_ZERO(&target);
    CPU_SET(cpu, &target);
    restore_ = pthread_setaffinity_np(
                   pthread_self(), sizeof(target), &target) == 0;
#else
    (void)cpu;
#endif
  }

  ~ScopedThreadAffinity() {
#if defined(__linux__)
    if (restore_)
      (void)pthread_setaffinity_np(
          pthread_self(), sizeof(saved_), &saved_);
#endif
  }

  ScopedThreadAffinity(const ScopedThreadAffinity &) = delete;
  ScopedThreadAffinity &operator=(const ScopedThreadAffinity &) = delete;

 private:
#if defined(__linux__)
  cpu_set_t saved_{};
  bool restore_ = false;
#endif
};

// Apply the configured real-time priority to the CURRENT thread. Call it once
// at each thread's entry. On failure (usually missing privilege) it warns once
// and leaves the thread at default scheduling, so the program keeps running
// (just without the RT guarantee). Success is logged once per role.
inline void ApplyThreadPriority(ThreadRole role, const char* name = "") {
#if defined(__linux__)
  const ThreadPriorityConfig& c = GetThreadPriorityConfig();
  if (!c.enable) return;

  sched_param sp;
  std::memset(&sp, 0, sizeof(sp));
  sp.sched_priority = PriorityForRole(role);

  const int rc = pthread_setschedparam(pthread_self(), c.policy, &sp);
  if (rc != 0) {
    static std::once_flag warned;
    std::call_once(warned, [rc] {
      std::cerr << "[prio] cannot set real-time priority (" << std::strerror(rc)
                << "). Grant CAP_SYS_NICE (e.g. `sudo setcap cap_sys_nice+ep "
                   "<binary>`), run as root, or raise rtprio in "
                   "/etc/security/limits.conf. Continuing with default "
                   "scheduling." << std::endl;
    });
    return;
  }

  static std::once_flag logged[3];
  const int idx = static_cast<int>(role);
  if (idx >= 0 && idx < 3) {
    std::call_once(logged[idx], [&c, name, &sp] {
      std::cout << "[prio] thread '" << name << "' policy="
                << (c.policy == SCHED_RR ? "RR" : "FIFO")
                << " prio=" << sp.sched_priority << std::endl;
    });
  }
#else
  (void)role;
  (void)name;
#endif
}

CYPERSTEREO_END_NAMESPACE

#endif  // CYPERSTEREO_THREAD_PRIORITY_H_
