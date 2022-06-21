#include "plant_selector/rosbag_panel.hpp"
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

#include <string.h>
#include <stdio.h>
#include <vector>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include <sensor_msgs/PointCloud2.h>

#include <math.h>
#include <cmath>

namespace rviz_custom_panel
{
    /**
     * Constructor of the panel, initializes member variables and creates the UI
     */
    RosbagPanel::RosbagPanel(QWidget * parent):rviz::Panel(parent) {
        pc_topic = "";

        // Construct and lay out labels and slider controls.
        QPushButton* sel_bag = new QPushButton("&Bag Select", this);
        pc_topic_edit = new QLineEdit(this);
        pc_topic_edit->setPlaceholderText("New Topic...");
        pc_topic_label = new QLabel("Chosen Topic: ");
        QLabel* frame_label = new QLabel("Frame");
        frame_slider = new QSlider(Qt::Horizontal);
        frame_slider->setMinimum(0);
        frame_slider->setMaximum(0);
        frame_number = new QLabel("");

        QGridLayout* controls_layout = new QGridLayout();
        controls_layout->addWidget(pc_topic_edit, 0, 0);
        controls_layout->addWidget(pc_topic_label, 0, 1);
        controls_layout->addWidget(sel_bag, 1, 0);
        controls_layout->addWidget(frame_label, 2, 0);
        controls_layout->addWidget(frame_slider, 2, 1);
        controls_layout->addWidget(frame_number, 2, 2);

        // Construct and lay out render panel.
        render_panel = new rviz::RenderPanel();
        QVBoxLayout* main_layout = new QVBoxLayout;
        main_layout->addLayout(controls_layout);

        // Set the top-level layout for this widget.
        setLayout(main_layout);
        frame_slider->setValue(0);

        set_zed_defaults();
 
        // Make signal/slot connections.
        connect(sel_bag, &QPushButton::clicked, this, &RosbagPanel::set_bag);
        connect(frame_slider, SIGNAL(valueChanged(int)), this, SLOT(set_frame(int)));
        connect(pc_topic_edit, SIGNAL(returnPressed()), this, SLOT(set_topic()));

        manager = new rviz::VisualizationManager(render_panel);
        manager->initialize();
        manager->startUpdate();

        frame_pub = n.advertise<sensor_msgs::PointCloud2>(pc_topic, 1);
    }

    void RosbagPanel::set_zed_defaults() {
        pc_topic = "/zed2i/zed_node/point_cloud/cloud_registered";
        pc_topic_label->setText(QString::fromUtf8(("Chosen Topic: " + pc_topic).c_str()));
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
        topics.push_back(pc_topic);

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // clear out any previous frames 
        frames.clear();

        // check if the topic exists in the rosbag
        bool does_exist = false;
        std::vector<const rosbag::ConnectionInfo *> connection_infos = view.getConnections();
        foreach(const rosbag::ConnectionInfo *info, connection_infos) {
            if(topics[0] == info->topic) {
                does_exist = true;
            }
        }

        if(!does_exist) {
            pc_topic_label->setText("Invalid Topic Name, Try Again.");
            return;
        }

        // Topic does exist, read in the data.
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

    void RosbagPanel::set_topic() {
        pc_topic = pc_topic_edit->text().toStdString();
        pc_topic_edit->setText("");
        pc_topic_label->setText(QString::fromUtf8(("Chosen Topic: " + pc_topic).c_str()));

        frame_pub = n.advertise<sensor_msgs::PointCloud2>(pc_topic, 1);
    }

} // namespace rviz_custom_panel

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_custom_panel::RosbagPanel, rviz::Panel)