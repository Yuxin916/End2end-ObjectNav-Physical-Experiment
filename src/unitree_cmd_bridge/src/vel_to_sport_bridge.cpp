#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <cstdio>   // std::snprintf
#include <cerrno>   // errno
#include <atomic>
#include <chrono>

#pragma pack(push, 1)
struct VelPacketV1 {
  char magic[4];     // "V2SB" (Vel-to-Sport-Bridge)
  uint8_t version;   // 1
  uint8_t reserved;  // 0
  uint16_t flags;    // bitfield reserved (0)
  uint32_t seq;      // increasing sequence number (network byte order)
  float vx;          // m/s  (+x forward)
  float vy;          // m/s  (+y left)
  float wz;          // rad/s (+yaw CCW)
};
#pragma pack(pop)

class VelToSportBridge : public rclcpp::Node {
public:
  VelToSportBridge() : Node("vel_to_sport_bridge") {
    // Parameters
    target_ip_   = this->declare_parameter<std::string>("target_ip", "127.0.0.1");
    target_port_ = this->declare_parameter<int>("target_port", 50051);
    use_csv_     = this->declare_parameter<bool>("use_csv", false); // if true, send "vx,vy,wz\n"
    qos_depth_   = this->declare_parameter<int>("qos_depth", 10);
    topic_       = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");

    // Create UDP socket
    sock_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd_ < 0) {
      RCLCPP_FATAL(get_logger(), "Failed to create UDP socket: %s", strerror(errno));
      throw std::runtime_error("socket");
    }
    std::memset(&dst_addr_, 0, sizeof(dst_addr_));
    dst_addr_.sin_family = AF_INET;
    dst_addr_.sin_port = htons(static_cast<uint16_t>(target_port_));
    if (::inet_pton(AF_INET, target_ip_.c_str(), &dst_addr_.sin_addr) != 1) {
      RCLCPP_FATAL(get_logger(), "Invalid target_ip '%s'", target_ip_.c_str());
      ::close(sock_fd_);
      throw std::runtime_error("inet_pton");
    }

    RCLCPP_INFO(get_logger(), "Forwarding '%s' -> UDP %s:%d (%s payload)",
                topic_.c_str(), target_ip_.c_str(), target_port_, use_csv_ ? "CSV" : "binary");

    // QoS: use sensor-data profile to keep up with fast publishers
    rclcpp::QoS qos(rclcpp::SensorDataQoS().keep_last(qos_depth_));
    sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        topic_, qos,
        std::bind(&VelToSportBridge::onTwist, this, std::placeholders::_1));
  }

  ~VelToSportBridge() override {
    if (sock_fd_ >= 0) {
      ::close(sock_fd_);
    }
  }

private:
  void onTwist(const geometry_msgs::msg::Twist::SharedPtr msg) {
    const float vx = static_cast<float>(msg->linear.x);
    const float vy = static_cast<float>(msg->linear.y);
    const float wz = static_cast<float>(msg->angular.z);

    ssize_t sent = 0;
    if (use_csv_) {
      // Simple text: "vx,vy,wz\n"
      char buf[128];
      int n = std::snprintf(buf, sizeof(buf), "%.6f,%.6f,%.6f\n", vx, vy, wz);
      sent = ::sendto(sock_fd_, buf, static_cast<size_t>(n), 0,
                      reinterpret_cast<sockaddr*>(&dst_addr_), sizeof(dst_addr_));
    } else {
      // Binary packet
      VelPacketV1 pkt{};
      std::memcpy(pkt.magic, "V2SB", 4);
      pkt.version = 1;
      pkt.reserved = 0;
      pkt.flags = 0;
      // keep seq monotonic even if callback is multi-threaded
      uint32_t s = seq_.fetch_add(1, std::memory_order_relaxed);
      pkt.seq = htonl(s);
      pkt.vx = vx;
      pkt.vy = vy;
      pkt.wz = wz;

      sent = ::sendto(sock_fd_, &pkt, sizeof(pkt), 0,
                      reinterpret_cast<sockaddr*>(&dst_addr_), sizeof(dst_addr_));
    }

    if (sent < 0) {
      // use throttled warning to avoid spamming logs
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000,
                           "sendto() failed: %s", strerror(errno));
    }
  }

  // Params
  std::string target_ip_;
  int target_port_;
  bool use_csv_;
  int qos_depth_;
  std::string topic_;

  // UDP
  int sock_fd_{-1};
  sockaddr_in dst_addr_{};

  // State
  std::atomic<uint32_t> seq_{0};
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelToSportBridge>());
  rclcpp::shutdown();
  return 0;
}
