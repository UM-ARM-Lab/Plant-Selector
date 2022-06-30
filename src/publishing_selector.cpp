// A lot of this code was referenced from https://github.com/drwnz/selected_points_publisher

#include "rviz/selection/selection_manager.h"
#include "rviz/viewport_mouse_event.h"
#include "rviz/display_context.h"
#include "rviz/selection/forwards.h"
#include "rviz/properties/property_tree_model.h"
#include "rviz/properties/property.h"
#include "rviz/properties/color_property.h"
#include "rviz/properties/vector_property.h"
#include "rviz/properties/float_property.h"
#include "rviz/view_manager.h"
#include "rviz/view_controller.h"
#include "OGRE/OgreCamera.h"

#include "plant_selector/publishing_selector.hpp"

#include <ros/ros.h>
#include <ros/time.h>
#include <sensor_msgs/PointCloud2.h>
#include <QVariant>
#include <visualization_msgs/Marker.h>
#include <boost/foreach.hpp>

namespace rviz_custom_tool
{
PublishingSelector::PublishingSelector() {
  updateTopic();
}

PublishingSelector::~PublishingSelector() {}

// TODO: Don't hardcode frame ids
void PublishingSelector::updateTopic() {
    rviz_cloud_topic = std::string("/rviz_selected_points");

    rviz_selected_publisher = n.advertise<sensor_msgs::PointCloud2>(rviz_cloud_topic, 1);
    is_selecting_pub = n.advertise<std_msgs::Bool>("/plant_selector/is_selecting", 1);
    instant_sub = n.subscribe("/plant_selector/is_instant", 1000, &PublishingSelector::instant_pub_handler, this);
    is_instant = true;

    ros::param::get("frame_id", frame_id);

    num_selected_points = 0;
}

void PublishingSelector::instant_pub_handler(const std_msgs::Bool::ConstPtr& msg) {
    is_instant = msg->data;
}

void PublishingSelector::clear_points() {
    rviz::SelectionManager* selection_manager = context_->getSelectionManager();
    rviz::M_Picked selection = selection_manager->getSelection();
    selection_manager->removeSelection(selection);
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = frame_id; 
    marker.header.stamp = ros::Time::now();
    marker.ns = "basic_shapes";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.lifetime = ros::Duration();
    num_selected_points = 0;
}

int PublishingSelector::processKeyEvent(QKeyEvent* event, rviz::RenderPanel* panel) {
    if (event->type() == QKeyEvent::KeyPress) {
        // clear points
        if (event->key() == 'c' || event->key() == 'C') {
            clear_points();
        }
        else if (event->key() == 'p' || event->key() == 'P') {
            rviz_selected_publisher.publish(selected_points);
            clear_points();
        }
    }

    // this return removes a warning, not sure what this func should return
    return 0;
}

int PublishingSelector::processMouseEvent(rviz::ViewportMouseEvent& event) {
    int flags = rviz::SelectionTool::processMouseEvent(event);
    if (event.alt()) {
        selecting = false;
        std_msgs::Bool msg;
        msg.data = selecting;
        is_selecting_pub.publish(msg);
    }
    else {
        if (event.leftDown()) {
            selecting = true;
            std_msgs::Bool msg;
            msg.data = selecting;
            is_selecting_pub.publish(msg);
        }
    }

    if (selecting) {
        if (event.leftUp()) {
            this->processSelectedArea();
            std_msgs::Bool msg;
            msg.data = false;
            is_selecting_pub.publish(msg);
        }
    }
    return flags;
}

int PublishingSelector::processSelectedArea() {
    rviz::SelectionManager* selection_manager = context_->getSelectionManager();
    rviz::M_Picked selection = selection_manager->getSelection();
    rviz::PropertyTreeModel* model = selection_manager->getPropertyModel();

    selected_points.header.frame_id = frame_id; 
    selected_points.height = 1;
    selected_points.point_step = 4 * 4;
    selected_points.is_dense = false;
    selected_points.is_bigendian = false;
    selected_points.fields.resize(4);

    selected_points.fields[0].name = "x";
    selected_points.fields[0].offset = 0;
    selected_points.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    selected_points.fields[0].count = 1;

    selected_points.fields[1].name = "y";
    selected_points.fields[1].offset = 4;
    selected_points.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    selected_points.fields[1].count = 1;

    selected_points.fields[2].name = "z";
    selected_points.fields[2].offset = 8;
    selected_points.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    selected_points.fields[2].count = 1;

    selected_points.fields[3].name = "rgb";
    selected_points.fields[3].offset = 12;
    selected_points.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    selected_points.fields[3].count = 1;


    int i = 0;
    while (model->hasIndex(i, 0)) {
        selected_points.row_step = (i + 1) * selected_points.point_step;
        selected_points.data.resize(selected_points.row_step);

        QModelIndex child_index = model->index(i, 0);

        rviz::Property* child = model->getProp(child_index);
        rviz::VectorProperty* subchild = (rviz::VectorProperty*)child->childAt(0);
        Ogre::Vector3 point_data = subchild->getVector();

        uint8_t* data_pointer = &selected_points.data[0] + i * selected_points.point_step;
        *(float*)data_pointer = point_data.x;
        data_pointer += 4;
        *(float*)data_pointer = point_data.y;
        data_pointer += 4;
        *(float*)data_pointer = point_data.z;
        data_pointer += 4;

        // Search for the rgb value
        for (int j = 1; j < child->numChildren(); j++) {
            rviz::Property* grandchild = child->childAt(j);
            QString nameOfChild = grandchild->getName();
            QString nameOfRgb("rgb");

            if (nameOfChild.contains(nameOfRgb)) {
                rviz::ColorProperty* colorchild = (rviz::ColorProperty*)grandchild;
                QColor thecolor = colorchild->getColor();

                int r, g, b;

                thecolor.getRgb(&r, &g, &b);
                int rgb = 0x00000000;
                rgb |= (0xff & r) << 16; 
                rgb |= (0xff & g) << 8;  
                rgb |= (0xff & b) << 0;

                // dont even ask me how this works, I lost so many brain cells here
                float x;
                *((int*)&x) = rgb;
                *(float*)data_pointer = x;
                break;
            }
        }
        data_pointer += 4;
        i++;
    }
    num_selected_points = i;

    selected_points.width = i;
    selected_points.header.stamp = ros::Time::now();

    if(is_instant) {
        rviz_selected_publisher.publish(selected_points);
        clear_points();
    }
    return 0;
}
}  // namespace rviz_custom_tool

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(rviz_custom_tool::PublishingSelector, rviz::Tool)
