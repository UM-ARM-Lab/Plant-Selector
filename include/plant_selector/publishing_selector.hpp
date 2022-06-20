// lot of code referenced from https://github.com/drwnz/selected_points_publisher

#ifndef SELECTED_POINTS_PUBLISHER_HPP
#define SELECTED_POINTS_PUBLISHER_HPP

#ifndef Q_MOC_RUN  // See: https://bugreports.qt-project.org/browse/QTBUG-22829
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include "rviz/tool.h"
#include <QCursor>
#include <QObject>
#endif

#include <sensor_msgs/PointCloud2.h>
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

public Q_SLOTS:
  void updateTopic();

protected:
  int processSelectedArea();
  ros::NodeHandle node_handle_;
  ros::Publisher rviz_selected_publisher_;
  ros::Subscriber pointcloud_subscriber_;

  std::string tf_frame_;
  std::string rviz_cloud_topic_;
  std::string subscribed_cloud_topic_;

  sensor_msgs::PointCloud2 selected_points_;

  bool selecting_;
  int num_selected_points_;
};
}  // namespace rviz_custom_tool

#endif  // SELECTED_POINTS_PUBLISHER_HPP
