#include "plant_selector/rosbag_panel.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <QColor>
#include <QSlider>
#include <QLabel>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPushButton>
#include <QLineEdit>

#include "rviz/visualization_manager.h"
#include "rviz/render_panel.h"
#include "rviz/display.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <interactive_markers/interactive_marker_server.h>

#include <string.h>
#include <stdio.h>
#include <vector>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include <sensor_msgs/PointCloud2.h>

#include <math.h>
#include <cmath>

#include "std_msgs/Bool.h"
#include "std_msgs/Float32MultiArray.h"

PLUGINLIB_EXPORT_CLASS(rviz_custom_panel::RosbagPanel, rviz::Panel)

namespace rviz_custom_panel
{
    /**
     * Constructor of the panel, initializes member variables and creates the UI
     */
    RosbagPanel::RosbagPanel(QWidget * parent):rviz::Panel(parent) {
        frame_pub = n.advertise<sensor_msgs::PointCloud2>("/camera/depth/color/points", 1);

        // Construct and lay out labels and slider controls.
        QPushButton* sel_bag = new QPushButton("&Bag Select", this);
        QLineEdit* choose_topic = new QLineEdit;
        choose_topic->setPlaceholderText("Enter topic of pointcloud to start publishing");
        QLabel* frame_label = new QLabel("Frame");
        frame_slider = new QSlider(Qt::Horizontal);
        frame_slider->setMinimum(0);
        frame_slider->setMaximum(0);
        frame_number = new QLabel("");

        QGridLayout* controls_layout = new QGridLayout();
        controls_layout->addWidget(sel_bag, 0, 0);
        controls_layout->addWidget(choose_topic, 0, 1);
        controls_layout->addWidget(frame_label, 1, 0);
        controls_layout->addWidget(frame_slider, 1, 1);
        controls_layout->addWidget(frame_number, 1, 2);

        // Construct and lay out render panel.
        render_panel = new rviz::RenderPanel();
        QVBoxLayout* main_layout = new QVBoxLayout;
        main_layout->addLayout(controls_layout);

        // Set the top-level layout for this widget.
        setLayout(main_layout);
        frame_slider->setValue(0);
 
        // Make signal/slot connections.
        connect(sel_bag, &QPushButton::clicked, this, &RosbagPanel::set_bag);
        connect(frame_slider, SIGNAL(valueChanged(int)), this, SLOT(set_frame(int)));

        manager = new rviz::VisualizationManager(render_panel);
        manager->initialize();
        manager->startUpdate();
    }

    /**
     *  Save all configuration data from this panel to the given
     *  Config object. It is important here that you call save()
     *  on the parent class so the class id and panel name get saved.
     */
    void RosbagPanel::save(rviz::Config config) const {
        rviz::Panel::save(config);
    }

    /**
     *  Load all configuration data for this panel from the given Config object.
     */
    void RosbagPanel::load(const rviz::Config & config) {
        rviz::Panel::load(config);
    }
    
    /**
     * After pressing "Bag Select", prompt the user with a file system to choose a bag, after that, load the first frame of the bag into rviz 
     */
    void RosbagPanel::set_bag() {
        // change the default filepath below, example: home/christianforeman/catkin_ws/src/point_cloud_selector" 
        std::string bag_filepath = QFileDialog::getOpenFileName(this, tr("Open Bag"), "/home/christianforeman/catkin_ws/src/plant_selector/bags", tr("Bags (*.bag)")).toStdString();
        // if its empty, return
        if(bag_filepath.empty()) {
            return;
        }

        // read in the rosbag
        rosbag::Bag bag;
        bag.open(bag_filepath);

        std::vector<std::string> topics;
        topics.push_back(std::string("/camera/depth/color/points"));

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // clear out any previous frames 
        frames.clear();

        foreach(rosbag::MessageInstance const m, view) {
            sensor_msgs::PointCloud2::ConstPtr temp = m.instantiate<sensor_msgs::PointCloud2>();
            
            frames.push_back(*temp);
        }
        
        bag.close();

        // publish first frame of the bag, set the range of the slider
        frame_pub.publish(frames[0]);
        frame_slider->setMaximum(frames.size() - 1);
        frame_number->setText("0");
    }
    
    /**
     * This function is called when the frame of the bag is changed on the slider, needs to wipe out any previous selections 
     */
    void RosbagPanel::set_frame(int new_frame_num) {
        frame_number->setText(QString::number(new_frame_num));
        frame_pub.publish(frames[new_frame_num]);
    }
} // namespace rviz_custom_panel
