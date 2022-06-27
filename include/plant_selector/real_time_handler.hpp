
#ifndef REAL_TIME_HANDLER_HPP 
#define REAL_TIME_HANDLER_HPP 

#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "sensor_msgs/PointCloud2.h"

class RealTimeHandler {
public:
    RealTimeHandler();
    void new_frame_handler(sensor_msgs::PointCloud2);
    void is_selecting_handler(std_msgs::Bool);

private:
// private variables
    ros::NodeHandle n;
    ros::Publisher pc_pub;
    ros::Subscriber pc_sub;
    ros::Subscriber is_selecting_sub;

    bool is_selecting; // if a selection is in progress, don't publish new frames
};

#endif