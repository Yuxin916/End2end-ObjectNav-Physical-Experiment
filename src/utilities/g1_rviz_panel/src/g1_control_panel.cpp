#include "g1_rviz_panel/g1_control_panel.hpp"

#include <QLabel>
#include <QPushButton>
#include <QSizePolicy>
#include <QVBoxLayout>

#include <rviz_common/display_context.hpp>

namespace g1_rviz_panel
{

G1ControlPanel::G1ControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout;
  root->setSpacing(6);
  root->setContentsMargins(6, 6, 6, 6);

  auto * title = new QLabel("G1 Controls");
  title->setStyleSheet("font-weight:bold; font-size:16px; padding:2px;");
  root->addWidget(title);

  // One button per row, full width -- dock the panel on the right and widen it.
  root->addWidget(makeButton("Big Wave", "wave_hi", "#3380e6"));
  root->addWidget(makeButton("Small Wave", "wave_lo", "#3380e6"));
  root->addWidget(makeButton("Clap", "clap", "#3380e6"));
  root->addWidget(makeButton("Reset Arm", "reset", "#73737f"));
  root->addWidget(makeButton("To Stage", "stage", "#f28c1a"));
  root->addWidget(makeButton("Backstage", "back", "#f28c1a"));
  root->addWidget(makeButton("Color ON", "blink_on", "#26bf40"));
  root->addWidget(makeButton("Color OFF", "blink_off", "#d93333"));
  root->addStretch();

  setLayout(root);
}

QPushButton * G1ControlPanel::makeButton(
  const QString & label, const QString & cmd, const char * color)
{
  auto * b = new QPushButton(label);
  b->setMinimumHeight(52);
  b->setCursor(Qt::PointingHandCursor);
  b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  b->setStyleSheet(
    QString(
      "QPushButton{background:%1; color:white; font-size:16px; font-weight:bold;"
      " border:none; border-radius:6px; padding:8px;}"
      "QPushButton:pressed{background:#1a1a1a;}")
    .arg(color));
  connect(b, &QPushButton::clicked, this, [this, cmd]() { sendCmd(cmd); });
  return b;
}

void G1ControlPanel::onInitialize()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
  pub_ = node_->create_publisher<std_msgs::msg::String>(
    "/g1_panel_cmd", rclcpp::QoS(10));
}

void G1ControlPanel::sendCmd(const QString & cmd)
{
  if (!pub_) {
    return;
  }
  std_msgs::msg::String msg;
  msg.data = cmd.toStdString();
  pub_->publish(msg);
}

}  // namespace g1_rviz_panel

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(g1_rviz_panel::G1ControlPanel, rviz_common::Panel)
