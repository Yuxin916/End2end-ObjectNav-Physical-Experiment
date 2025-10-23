#include <chrono>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

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

    // Build a concrete QoS (avoid ternary type mismatch)
    rclcpp::QoS qos = rclcpp::SystemDefaultsQoS();
    if (sensor_qos_) qos = rclcpp::SensorDataQoS();

    // Subscribe to BOTH types on the same topic
    sub_twist_ = create_subscription<geometry_msgs::msg::Twist>(
      topic_, qos, [this](geometry_msgs::msg::Twist::ConstSharedPtr msg){
        std::lock_guard<std::mutex> lk(mtx_);
        last_cmd_.linear  = msg->linear;
        last_cmd_.angular = msg->angular;
        last_time_        = now();
        have_cmd_         = true;
      });

    sub_twist_stamped_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      topic_, qos, [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg){
        std::lock_guard<std::mutex> lk(mtx_);
        last_cmd_.linear  = msg->twist.linear;
        last_cmd_.angular = msg->twist.angular;
        last_time_        = now();
        have_cmd_         = true;
      });

    const auto period = std::chrono::duration<double>(1.0 / pub_hz_);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VelToSportBridge::tick, this));

    // Initialize SportClient; disable joystick passthrough if needed
    try {
      tc_.Init();
      // Optional: tc_.SwitchJoystick(false);  // ensure external control allowed
      // Optional: tc_.StandUp();              // if robot needs to stand before accepting Move
    } catch (const std::exception &e) {
      RCLCPP_WARN(get_logger(), "SportClient Init threw: %s", e.what());
    }

    RCLCPP_INFO(get_logger(),
      "vel_to_sport_bridge on %s (Twist/TwistStamped), hz=%.1f, timeout=%.2fs",
      topic_.c_str(), pub_hz_, timeout_s_);
  }

  ~VelToSportBridge() override {
    try { (void)send_to_sdk_(0.0, 0.0, 0.0); (void)tc_.StopMove(); } catch (...) {}
  }

private:
  void tick() {
    geometry_msgs::msg::Twist cmd;
    bool active = false;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      const auto age = (now() - last_time_).seconds();
      if (have_cmd_ && age <= timeout_s_) { cmd = last_cmd_; active = true; }
    }

    double vx = clamp_(cmd.linear.x,  -max_vx_, max_vx_);
    double vy = clamp_(cmd.linear.y,  -max_vy_, max_vy_);
    double wz = clamp_(cmd.angular.z, -max_wz_, max_wz_);

    if (!active) { vx = vy = wz = 0.0; }

    if (invert_vx_) vx = -vx;
    if (invert_vy_) vy = -vy;
    if (invert_wz_) wz = -wz;

    if (cap_ellipse_) {
      const double nx = std::abs(vx) / (max_vx_ > 1e-6 ? max_vx_ : 1e-6);
      const double ny = std::abs(vy) / (max_vy_ > 1e-6 ? max_vy_ : 1e-6);
      const double r  = std::hypot(nx, ny);
      if (r > 1.0) { const double s = 1.0 / r; vx *= s; vy *= s; }
    }

    try {
      int32_t ret = send_to_sdk_(vx, vy, wz);   // uses SportClient::Move
      if (ret != 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "SportClient returned nonzero (%d).", ret);
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "SportClient call threw: %s", e.what());
    }
  }

  inline int32_t send_to_sdk_(double vx, double vy, double wz) {
    return tc_.Move(static_cast<float>(vx), static_cast<float>(vy), static_cast<float>(wz));
  }

  static double clamp_(double v, double lo, double hi){ return std::max(lo, std::min(v, hi)); }
  rclcpp::Time now() { return this->get_clock()->now(); }  // non-const (Humble)

  // Params/state
  std::string topic_;
  double timeout_s_{0.3};
  double pub_hz_{50.0};
  double max_vx_{0.8}, max_vy_{0.5}, max_wz_{1.5};
  bool cap_ellipse_{true}, invert_vx_{false}, invert_vy_{false}, invert_wz_{false}, sensor_qos_{false};

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr        sub_twist_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr sub_twist_stamped_;
  rclcpp::TimerBase::SharedPtr timer_;

  unitree::robot::go2::SportClient tc_;

  std::mutex mtx_;
  geometry_msgs::msg::Twist last_cmd_;
  rclcpp::Time last_time_{0, 0, RCL_ROS_TIME};
  bool have_cmd_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelToSportBridge>());
  rclcpp::shutdown();
  return 0;
}
