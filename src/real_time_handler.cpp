#include "plant_selector/real_time_handler.hpp"
#include "stdio.h"
#include "ros/console.h"
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "sensor_msgs/PointCloud2.h"

RealTimeHandler::RealTimeHandler() {
    pc_pub = n.advertise<sensor_msgs::PointCloud2>("/plant_selector/controlled_point_cloud", 1);
    // eventually this subscriber topic shouldn't be hardcoded.
    pc_sub = n.subscribe("/zed2i/zed_node/point_cloud/cloud_registered", 1000, &RealTimeHandler::new_frame_handler, this);
    is_selecting_sub = n.subscribe("/plant_selector/is_selecting", 1000, &RealTimeHandler::is_selecting_handler, this);

    is_selecting = false;
}

void RealTimeHandler::new_frame_handler(sensor_msgs::PointCloud2 msg) {
    if(is_selecting) {
        return;
    }

    pc_pub.publish(msg);
}

void RealTimeHandler::is_selecting_handler(std_msgs::Bool msg) {
    is_selecting = msg.data; 
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "real_time_handler");
    RealTimeHandler handler = RealTimeHandler();
    ros::spin();
    return 0;
}