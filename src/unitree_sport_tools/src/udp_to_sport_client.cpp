#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

// Unitree SDK2
#include <unitree/robot/go2/sport/sport_client.hpp>
using unitree::robot::go2::SportClient;

#pragma pack(push, 1)
struct VelPacketV1 {
  char magic[4];     // "V2SB"
  uint8_t version;   // 1
  uint8_t reserved;  // 0
  uint16_t flags;    // 0
  uint32_t seq;      // network byte order
  float vx;          // m/s (+x forward)
  float vy;          // m/s (+y left)
  float wz;          // rad/s (+yaw CCW)
};
#pragma pack(pop)

static inline double clamp(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

int main(int, char**) {
  // --- params via env or defaults ---
  const char* env_port   = std::getenv("VEL_UDP_PORT");
  const char* env_csv    = std::getenv("VEL_USE_CSV");        // "1" => CSV mode
  const char* env_iface  = std::getenv("VEL_BIND_ADDR");      // e.g. "0.0.0.0"
  const char* env_deadms = std::getenv("VEL_DEADMAN_MS");     // e.g. "300"
  const char* env_rate   = std::getenv("VEL_CTRL_HZ");        // e.g. "100"
  const char* env_lim    = std::getenv("VEL_LIMITS");         // "vx,vy,wz" e.g. "1.0,0.6,2.0"
  const bool  DISABLE_SDK = (std::getenv("DISABLE_SDK") && std::string(std::getenv("DISABLE_SDK"))=="1");

  const int   PORT         = env_port   ? std::atoi(env_port) : 50051;
  const bool  USE_CSV      = env_csv    ? (std::string(env_csv)=="1") : false;
  const char* BIND_ADDR    = env_iface  ? env_iface : "0.0.0.0";
  const int   DEADMAN_MS   = env_deadms ? std::atoi(env_deadms) : 300;
  const int   CTRL_HZ      = env_rate   ? std::atoi(env_rate) : 100;
  double VX_MAX = 1.0, VY_MAX = 0.6, WZ_MAX = 2.0;
  if (env_lim) {
    double a,b,c; if (std::sscanf(env_lim, "%lf,%lf,%lf", &a,&b,&c) == 3) {
      VX_MAX = std::fabs(a); VY_MAX = std::fabs(b); WZ_MAX = std::fabs(c);
    }
  }

  // --- UDP socket ---
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) { perror("socket"); return 1; }
  int yes = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)PORT);
  if (::inet_pton(AF_INET, BIND_ADDR, &addr.sin_addr) != 1) {
    std::fprintf(stderr, "Invalid VEL_BIND_ADDR: %s\n", BIND_ADDR); return 1;
  }
  if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

  std::printf("[udp_to_sport_client] listening on %s:%d (%s)%s\n",
              BIND_ADDR, PORT, USE_CSV ? "CSV" : "binary",
              DISABLE_SDK ? " [DISABLE_SDK=1]" : "");

  // --- State & timing ---
  std::atomic<float> last_vx{0.f}, last_vy{0.f}, last_wz{0.f};
  std::atomic<int64_t> last_ms{0};
  std::atomic<bool> running{true};
  auto now_ms = [](){
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
  };
  last_ms.store(now_ms());

  // --- Lazy SDK client (constructed on first packet) ---
  std::shared_ptr<SportClient> tc;

  // --- receive thread: updates last_vx/vy/wz + timestamp ---
  std::thread rx([&](){
    uint8_t buf[256];
    while (running.load()) {
      sockaddr_in src{}; socklen_t slen = sizeof(src);
      ssize_t n = ::recvfrom(fd, buf, sizeof(buf), 0, (sockaddr*)&src, &slen);
      if (n < 0) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

      float vx=0, vy=0, wz=0; bool ok=false;
      if (USE_CSV) {
        buf[std::min<ssize_t>(n, 255)] = 0;
        double dvx,dvy,dwz;
        if (std::sscanf((char*)buf, "%lf,%lf,%lf", &dvx,&dvy,&dwz) == 3) {
          vx=dvx; vy=dvy; wz=dwz; ok=true;
        }
      } else if (n >= (ssize_t)sizeof(VelPacketV1)) {
        VelPacketV1 pkt{}; std::memcpy(&pkt, buf, sizeof(VelPacketV1));
        if (std::memcmp(pkt.magic, "V2SB", 4)==0 && pkt.version==1) {
          vx=pkt.vx; vy=pkt.vy; wz=pkt.wz; ok=true;
        }
      }

      if (ok) {
        vx = (float)clamp(vx, -VX_MAX, VX_MAX);
        vy = (float)clamp(vy, -VY_MAX, VY_MAX);
        wz = (float)clamp(wz, -WZ_MAX, WZ_MAX);

        last_vx.store(vx, std::memory_order_relaxed);
        last_vy.store(vy, std::memory_order_relaxed);
        last_wz.store(wz, std::memory_order_relaxed);
        last_ms.store(now_ms(), std::memory_order_relaxed);

        if (!DISABLE_SDK && !tc) {
          std::printf("[udp_to_sport_client] creating SportClient on first packet...\n");
          // Construct here—if it crashes, the crash point is clear.
          tc = std::make_shared<SportClient>();
          std::printf("[udp_to_sport_client] SportClient created.\n");
        }
      }
    }
  });

  // --- control loop: maintain cadence + deadman ---
  const double dt = 1.0 / std::max(1, CTRL_HZ);
  int tick=0;
  while (running.load()) {
    const int64_t tnow = now_ms();
    float vx = last_vx.load(std::memory_order_relaxed);
    float vy = last_vy.load(std::memory_order_relaxed);
    float wz = last_wz.load(std::memory_order_relaxed);

    if (tnow - last_ms.load(std::memory_order_relaxed) > DEADMAN_MS) { vx=vy=wz=0.f; }

    if (DISABLE_SDK) {
      if ((tick++ % (CTRL_HZ*2))==0) {
        std::printf("[udp_to_sport_client] (DRY) vx=%.3f vy=%.3f wz=%.3f\n", vx,vy,wz);
      }
    } else if (tc) {
      try {
        tc->Move(vx, vy, wz);
      } catch (...) {
        // swallow occasional transport errors; keep looping
      }
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
  }

  running.store(false);
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
  rx.join();

  try { if (tc) tc->Move(0.f,0.f,0.f); } catch (...) {}
  return 0;
}
