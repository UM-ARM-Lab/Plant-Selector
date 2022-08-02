// lot of code referenced from https://github.com/drwnz/selected_points_publisher

#ifndef SELECTED_POINTS_PUBLISHER_HPP
#define SELECTED_POINTS_PUBLISHER_HPP

#ifndef Q_MOC_RUN  // See: https://bugreports.qt-project.org/browse/QTBUG-22829
#include <ros/ros.h>
#include "rviz/tool.h"
#include <QCursor>
#include <QObject>
#endif

#include <sensor_msgs/PointCloud2.h>
#include "std_msgs/Bool.h"
#include "rviz/default_plugin/tools/selection_tool.h"

namespace rviz_custom_tool
{
class PublishingSelector;

class PublishingSelector: public rviz::SelectionTool
{
  Q_OBJECT
public:
  PublishingSelector();
  virtual ~PublishingSelector();
  virtual int processMouseEvent(rviz::ViewportMouseEvent& event);
  virtual int processKeyEvent(QKeyEvent* event, rviz::RenderPanel* panel);
  void clear_points();

public Q_SLOTS:
  void updateTopic();
  void instant_pub_handler(const std_msgs::Bool::ConstPtr& msg);

protected:
  int processSelectedArea();
  ros::NodeHandle n;
  ros::Publisher rviz_selected_pub;
  ros::Publisher is_selecting_pub;
  ros::Subscriber instant_sub;

  std::string tf_frame;
  std::string rviz_cloud_topic;
  std::string frame_id;

  sensor_msgs::PointCloud2 selected_points;

  bool is_instant;
  bool selecting;
  int num_selected_points;
};
}  // namespace rviz_custom_tool

#endif  // SELECTED_POINTS_PUBLISHER_HPP
