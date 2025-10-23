#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstring>

// Unitree SDK (adjust include if yours differs)
#include "unitree/robot/go2/sport/sport_client.hpp"


static const char* CLIENT_SOCK = "/tmp/cmdvel_client.sock";
static volatile bool running = true;

void on_sig(int){ running = false; }

int main() {
  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);

  // UNIX dgram socket: bind to /tmp/cmdvel_client.sock
  int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
  if (fd < 0) { perror("socket"); return 1; }

  ::unlink(CLIENT_SOCK);
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, CLIENT_SOCK, sizeof(addr.sun_path)-1);
  if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

  // Unitree client (no ROS in this process)
  unitree::robot::go2::SportClient client(/*highlevel=*/true);

  // Optional: stand once connected (ignore errors if not ready yet)
  try { client.StandUp(); } catch(...) {}

  // Limits (tune to match your robot/safety)
  const double MAX_VX = 0.8, MAX_VY = 0.5, MAX_WZ = 1.5;

  while (running) {
    double buf[3];
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n != (ssize_t)sizeof(buf)) continue;

    double vx = buf[0], vy = buf[1], wz = buf[2];
    // clamp
    if (vx >  MAX_VX) vx =  MAX_VX; if (vx < -MAX_VX) vx = -MAX_VX;
    if (vy >  MAX_VY) vy =  MAX_VY; if (vy < -MAX_VY) vy = -MAX_VY;
    if (wz >  MAX_WZ) wz =  MAX_WZ; if (wz < -MAX_WZ) wz = -MAX_WZ;

    try { client.Move(vx, vy, wz); } catch(...) {}
  }

  try { client.StandUp(); } catch(...) {}
  ::close(fd);
  ::unlink(CLIENT_SOCK);
  return 0;
}
