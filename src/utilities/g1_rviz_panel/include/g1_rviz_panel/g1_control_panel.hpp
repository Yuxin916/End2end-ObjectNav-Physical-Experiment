#ifndef G1_RVIZ_PANEL__G1_CONTROL_PANEL_HPP_
#define G1_RVIZ_PANEL__G1_CONTROL_PANEL_HPP_

#include <QString>
#include <QWidget>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/string.hpp>

class QPushButton;

namespace g1_rviz_panel
{

// Docked RViz panel: one button per G1 action. Clicking publishes a short
// command string on /g1_panel_cmd; the unitree_control node dispatches it to
// the same action as the joystick. Kept intentionally dumb (fire-and-forget
// publisher) so it never blocks the RViz UI thread.
class G1ControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit G1ControlPanel(QWidget * parent = nullptr);

  void onInitialize() override;

private Q_SLOTS:
  void sendCmd(const QString & cmd);

private:
  QPushButton * makeButton(const QString & label, const QString & cmd, const char * color);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

}  // namespace g1_rviz_panel

#endif  // G1_RVIZ_PANEL__G1_CONTROL_PANEL_HPP_
