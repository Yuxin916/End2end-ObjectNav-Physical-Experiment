#include "g1_rviz_panel/g1_control_panel.hpp"

#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <rviz_common/display_context.hpp>

namespace g1_rviz_panel
{

G1ControlPanel::G1ControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout;

  auto * title = new QLabel("G1 Controls");
  title->setStyleSheet("font-weight:bold; font-size:15px; padding:2px;");
  root->addWidget(title);

  auto * grid = new QGridLayout;
  grid->setSpacing(4);
  // Gestures (blue).
  grid->addWidget(makeButton("Big Wave", "wave_hi", "#3380e6"), 0, 0);
  grid->addWidget(makeButton("Small Wave", "wave_lo", "#3380e6"), 0, 1);
  grid->addWidget(makeButton("Clap", "clap", "#3380e6"), 1, 0);
  grid->addWidget(makeButton("Reset Arm", "reset", "#73737f"), 1, 1);
  // Navigation (orange).
  grid->addWidget(makeButton("To Stage", "stage", "#f28c1a"), 2, 0, 1, 2);
  grid->addWidget(makeButton("Backstage", "back", "#f28c1a"), 3, 0, 1, 2);
  // Head LED (green / red).
  grid->addWidget(makeButton("Color ON", "blink_on", "#26bf40"), 4, 0);
  grid->addWidget(makeButton("Color OFF", "blink_off", "#d93333"), 4, 1);
  root->addLayout(grid);
  root->addStretch();

  setLayout(root);
}

QPushButton * G1ControlPanel::makeButton(
  const QString & label, const QString & cmd, const char * color)
{
  auto * b = new QPushButton(label);
  b->setMinimumHeight(42);
  b->setCursor(Qt::PointingHandCursor);
  b->setStyleSheet(
    QString(
      "QPushButton{background:%1; color:white; font-size:14px; font-weight:bold;"
      " border:none; border-radius:5px; padding:6px;}"
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
