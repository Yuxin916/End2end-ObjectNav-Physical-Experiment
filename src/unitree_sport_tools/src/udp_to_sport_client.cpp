#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>     // std::fprintf, std::sscanf, std::printf
#include <cstring>
#include <algorithm>  // std::min, std::max
#include <string>
#include <thread>
#include <vector>

// ---- Unitree SDK2 ----
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

using unitree::robot::go2::SportClient;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;
using unitree_go::msg::dds_::SportModeState_;

#define TOPIC_HIGHSTATE "rt/sportmodestate"

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

int main(int argc, char** argv) {
  // -----------------------------
  // Args / env
  // -----------------------------
  const char* env_port   = std::getenv("VEL_UDP_PORT");
  const char* env_csv    = std::getenv("VEL_USE_CSV");        // "1" => CSV mode
  const char* env_iface  = std::getenv("VEL_BIND_ADDR");      // e.g. "0.0.0.0"
  const char* env_deadms = std::getenv("VEL_DEADMAN_MS");     // e.g. "300"
  const char* env_rate   = std::getenv("VEL_CTRL_HZ");        // e.g. "100"
  const char* env_lim    = std::getenv("VEL_LIMITS");         // "vx,vy,wz"
  const char* env_nic    = std::getenv("VEL_NIC");            // fallback NIC if no argv[1]
  const bool  DISABLE_SDK = (std::getenv("DISABLE_SDK") && std::string(std::getenv("DISABLE_SDK"))=="1");

  const int   PORT         = env_port   ? std::atoi(env_port) : 50051;
  const bool  USE_CSV      = env_csv    ? (std::string(env_csv)=="1") : false;
  const char* BIND_ADDR    = env_iface  ? env_iface : "0.0.0.0";
  const int   DEADMAN_MS   = env_deadms ? std::atoi(env_deadms) : 300;
  const int   CTRL_HZ      = env_rate   ? std::atoi(env_rate) : 100;

  double VX_MAX = 1.0, VY_MAX = 0.6, WZ_MAX = 2.0;
  if (env_lim) {
    double a,b,c;
    if (std::sscanf(env_lim, "%lf,%lf,%lf", &a,&b,&c) == 3) {
      VX_MAX = std::abs(a); VY_MAX = std::abs(b); WZ_MAX = std::abs(c);
    }
  }

  // Determine robot NIC for Unitree DDS channel (matches Unitree samples)
  std::string nic;
  if (argc >= 2)      nic = argv[1];
  else if (env_nic)   nic = env_nic;
  else                nic = "eth0"; // sensible default on Go2 Jetson

  // -----------------------------
  // UDP socket
  // -----------------------------
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) { perror("socket"); return 1; }
  int yes = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)PORT);
  if (::inet_pton(AF_INET, BIND_ADDR, &addr.sin_addr) != 1) {
    std::fprintf(stderr, "Invalid VEL_BIND_ADDR: %s\n", BIND_ADDR); return 1;
  }
  if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

  std::printf("[udp_to_sport_client] listening on %s:%d (%s)%s  [NIC=%s]\n",
              BIND_ADDR, PORT, USE_CSV ? "CSV" : "binary",
              DISABLE_SDK ? " [DISABLE_SDK=1]" : "", nic.c_str());

  // -----------------------------
  // State & timing
  // -----------------------------
  std::atomic<float> last_vx{0.f}, last_vy{0.f}, last_wz{0.f};
  std::atomic<int64_t> last_ms{0};
  std::atomic<bool> running{true};
  auto now_ms = [](){
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
  };
  last_ms.store(now_ms());

  // -----------------------------
  // Unitree SDK bring-up (main thread), matching official sample
  // -----------------------------
  std::unique_ptr<SportClient> sport_client;
  ChannelSubscriberPtr<SportModeState_> suber;
  SportModeState_ state{};

  if (!DISABLE_SDK) {
    // Init DDS channel first
    unitree::robot::ChannelFactory::Instance()->Init(0, nic.c_str());

    // Create SportClient, set timeout, init
    sport_client = std::make_unique<SportClient>();
    sport_client->SetTimeout(10.0f);
    sport_client->Init();

    // Optional: subscribe to sport mode state (debugging/health)
    suber.reset(new ChannelSubscriber<SportModeState_>(TOPIC_HIGHSTATE));
    suber->InitChannel(
      [&](const void* msg){
        state = *(const SportModeState_*)msg;
        // // Uncomment for debugging:
        // std::printf("[state] pos=(%.3f, %.3f) yaw=%.3f\n",
        //             state.position()[0], state.position()[1], state.imu_state().rpy()[2]);
      }, 1);
  }

  // -----------------------------
  // RX thread (updates latest twist)
  // -----------------------------
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
      }
    }
  });

  // -----------------------------
  // Control loop (like sample’s recurrent thread)
  // -----------------------------
  const double dt = 1.0 / std::max(1, CTRL_HZ);
  int tick = 0;

  while (running.load()) {
    const int64_t tnow = now_ms();

    float vx = last_vx.load(std::memory_order_relaxed);
    float vy = last_vy.load(std::memory_order_relaxed);
    float wz = last_wz.load(std::memory_order_relaxed);

    // dead-man
    if (tnow - last_ms.load(std::memory_order_relaxed) > DEADMAN_MS) {
      vx = 0.f; vy = 0.f; wz = 0.f;
    }

    if (DISABLE_SDK) {
      if ((tick++ % (CTRL_HZ*2))==0) {
        std::printf("[udp_to_sport_client] (DRY) vx=%.3f vy=%.3f wz=%.3f\n", vx, vy, wz);
      }
    } else if (sport_client) {
      try {
        // Exactly like the official “velocity_move” case:
        sport_client->Move(vx, vy, wz);
      } catch (...) {
        // swallow sporadic transport errors; keep looping
      }
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(dt));
  }

  // -----------------------------
  // Shutdown
  // -----------------------------
  running.store(false);
  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);
  rx.join();

  try { if (sport_client) sport_client->Move(0.f,0.f,0.f); } catch (...) {}
  return 0;
}
