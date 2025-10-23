#include <chrono>
#include <memory>
#include <atomic>
#include <mutex>          // <-- needed
#include <algorithm>      // <-- for std::max/min

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"   // <-- needed

// Unitree SDK2 (Go2)
#include <unitree/robot/go2/sport/sport_client.hpp>

using namespace std::chrono_literals;

class VelToSportBridge : public rclcpp::Node {
public:
  VelToSportBridge() : rclcpp::Node("vel_to_sport_bridge") {
    topic_          = declare_parameter<std::string>("cmd_topic", "/cmd_vel");
    timeout_s_      = declare_parameter<double>("timeout", 0.3);
    pub_hz_         = declare_parameter<double>("publish_hz", 50.0);
    max_vx_         = declare_parameter<double>("max_vx", 0.8);
    max_vy_         = declare_parameter<double>("max_vy", 0.5);
    max_wz_         = declare_parameter<double>("max_wz", 1.5);
    cap_ellipse_    = declare_parameter<bool>("elliptical_cap", true);
    invert_vx_      = declare_parameter<bool>("invert_vx", false);
    invert_vy_      = declare_parameter<bool>("invert_vy", false);
    invert_wz_      = declare_parameter<bool>("invert_wz", false);
    sensor_qos_     = declare_parameter<bool>("use_sensor_qos", false);

    if (pub_hz_ < 1.0)  pub_hz_ = 1.0;
    if (pub_hz_ > 200.) pub_hz_ = 200.;

    auto qos = sensor_qos_ ? rclcpp::SensorDataQoS() : rclcpp::SystemDefaultsQoS();

    // Subscribe to BOTH; only the matching type will receive messages.
    sub_twist_ = create_subscription<geometry_msgs::msg::Twist>(
      topic_, qos,
      [this](geometry_msgs::msg::Twist::ConstSharedPtr msg){
        std::lock_guard<std::mutex> lk(mtx_);
        last_cmd_.linear   = msg->linear;
        last_cmd_.angular  = msg->angular;
        last_time_         = this->now();
        have_cmd_          = true;
      });

    sub_twist_stamped_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      topic_, qos,
      [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg){
        std::lock_guard<std::mutex> lk(mtx_);
        last_cmd_.linear   = msg->twist.linear;
        last_cmd_.angular  = msg->twist.angular;
        last_time_         = this->now();
        have_cmd_          = true;
      });

    const auto period = std::chrono::duration<double>(1.0 / pub_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VelToSportBridge::tick, this));

    RCLCPP_INFO(get_logger(),
      "vel_to_sport_bridge listening on %s (Twist or TwistStamped), hz=%.1f, timeout=%.2fs",
      topic_.c_str(), pub_hz_, timeout_s_);
  }

  ~VelToSportBridge() override {
    try { send_to_sdk_(0.0, 0.0, 0.0); } catch (...) {}
  }

private:
  void tick() {
    geometry_msgs::msg::Twist cmd;
    bool active = false;

    {
      std::lock_guard<std::mutex> lk(mtx_);
      const auto age = (now() - last_time_).seconds();
      if (have_cmd_ && age <= timeout_s_) {
        cmd = last_cmd_;
        active = true;
      }
    }

    double vx = clamp_(cmd.linear.x,  -max_vx_, max_vx_);
    double vy = clamp_(cmd.linear.y,  -max_vy_, max_vy_);
    double wz = clamp_(cmd.angular.z, -max_wz_, max_wz_);

    if (!active) { vx = 0.0; vy = 0.0; wz = 0.0; }

    // Optional sign flips (if your body-frame signs differ)
    if (invert_vx_) vx = -vx;
    if (invert_vy_) vy = -vy;
    if (invert_wz_) wz = -wz;

    // Optional elliptical cap in XY so diagonals don't exceed axis limits
    if (cap_ellipse_) {
      const double nx = std::abs(vx) / (max_vx_ > 1e-6 ? max_vx_ : 1e-6);
      const double ny = std::abs(vy) / (max_vy_ > 1e-6 ? max_vy_ : 1e-6);
      const double r  = std::hypot(nx, ny);
      if (r > 1.0) {
        const double s = 1.0 / r;
        vx *= s; vy *= s;
      }
    }

    // Send to Unitree SDK2 SportClient.
    int32_t ret = 0;
    try {
      ret = tc_.velocityCommand(vx, vy, wz);
      // If your SDK uses a different method name, swap the line above for one of these:
      // ret = tc_.move(vx, vy, wz);
      // ret = tc_.setBodyVelocity(vx, vy, wz);
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "SportClient send failed: %s", e.what());
    }
    (void)ret;
  }

  static double clamp_(double v, double lo, double hi){ return std::max(lo, std::min(v, hi)); }
  rclcpp::Time now() const { return get_clock()->now(); }

  // ---- Params/state ----
  std::string topic_;
  double timeout_s_{0.3};
  double pub_hz_{50.0};
  double max_vx_{0.8}, max_vy_{0.5}, max_wz_{1.5};
  bool cap_ellipse_{true}, invert_vx_{false}, invert_vy_{false}, invert_wz_{false}, sensor_qos_{false};

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_twist_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_stamped_;
  rclcpp::TimerBase::SharedPtr timer_;

  unitree::robot::go2::SportClient tc_;

  std::mutex mtx_;
  geometry_msgs::msg::Twist last_cmd_;
  rclcpp::Time last_time_{0, 0, RCL_ROS_TIME};   // safer init with clock type
  bool have_cmd_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelToSportBridge>());
  rclcpp::shutdown();
  return 0;
}
