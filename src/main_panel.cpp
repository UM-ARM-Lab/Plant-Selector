#include "plant_selector/main_panel.hpp"
#include <QColor>
#include <QSlider>
#include <QLabel>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <QLineEdit>

#include "rviz/visualization_manager.h"
#include "rviz/render_panel.h"
#include "rviz/display.h"

#include <string.h>
#include <stdio.h>
#include <vector>

#include <sensor_msgs/PointCloud2.h>

#include "std_msgs/String.h"
#include "std_msgs/Bool.h"

namespace rviz_custom_panel
{
    /**
     * Constructor of the panel, initializes member variables and creates the UI
     */
    MainPanel::MainPanel(QWidget * parent):rviz::Panel(parent) {
        // setup ros connections
        mode_pub = n.advertise<std_msgs::String>("/plant_selector/mode", 1);
        publish_time_pub = n.advertise<std_msgs::Bool>("/plant_selector/is_instant", 1);
        verification_pub = n.advertise<std_msgs::Bool>("/plant_selector/verification", 1);
        hide_gripper_pub = n.advertise<std_msgs::Bool>("/plant_selector/hide_gripper", 1);
        ask_verification_sub = n.subscribe("/plant_selector/ask_for_verification", 1000, &MainPanel::verification_callback, this);

        QLabel* publishing_label = new QLabel("Instantly Publish Selection?", this);

        QStringList pub_commands = {"Yes", "No"};
        QComboBox* pub_combo = new QComboBox(this);
        pub_combo->addItems(pub_commands);

        QLabel* inter_label = new QLabel("Interaction Type:", this);

        QStringList commands = {"Weed", "Branch"};
        QComboBox* combo = new QComboBox(this);
        combo->addItems(commands);

        verification_label = new QLabel("", this);
        yes_button = new QPushButton("&Yes", this);
        yes_button->setEnabled(false);
        no_button = new QPushButton("&No", this);
        no_button->setEnabled(false);

        QPushButton* hide_gripper_button = new QPushButton("&Hide Red Gripper", this);

        QGridLayout* controls_layout = new QGridLayout();
        controls_layout->addWidget(publishing_label, 0, 0);
        controls_layout->addWidget(pub_combo, 0, 1);
        controls_layout->addWidget(inter_label, 1, 0);
        controls_layout->addWidget(combo, 1, 1);
        controls_layout->addWidget(hide_gripper_button, 1, 2);
        controls_layout->addWidget(verification_label, 2, 0);
        controls_layout->addWidget(yes_button, 2, 1);
        controls_layout->addWidget(no_button, 2, 2);

        // Construct and lay out render panel.
        render_panel = new rviz::RenderPanel();
        QVBoxLayout* main_layout = new QVBoxLayout;
        main_layout->addLayout(controls_layout);

        // Set the top-level layout for this widget.
        setLayout(main_layout);
        
        // Make signal/slot connections.
        connect(pub_combo, &QComboBox::currentTextChanged, this, &MainPanel::publish_time_handler);
        connect(combo, &QComboBox::currentTextChanged, this, &MainPanel::extract_type_handler);
        connect(yes_button, &QPushButton::clicked, this, &MainPanel::yes_button_handler);
        connect(no_button, &QPushButton::clicked, this, &MainPanel::no_button_handler);
        connect(hide_gripper_button, &QPushButton::clicked, this, &MainPanel::hide_gripper_handler);

        manager = new rviz::VisualizationManager(render_panel);
        manager->initialize();
        manager->startUpdate();
    }

    /**
     *  Save all configuration data from this panel to the given
     *  Config object. It is important here that you call save()
     *  on the parent class so the class id and panel name get saved.
     */
    void MainPanel::save(rviz::Config config) const {
        rviz::Panel::save(config);
    }

    /**
     *  Load all configuration data for this panel from the given Config object.
     */
    void MainPanel::load(const rviz::Config & config) {
        rviz::Panel::load(config);
    }

    void MainPanel::verification_callback(const std_msgs::Bool::ConstPtr& msg) {
        if(msg->data) {
            verification_label->setText("Execute this Plan?");
            yes_button->setEnabled(true);
            no_button->setEnabled(true);
        }
        else {
            verification_label->setText("No Valid Plan");
            yes_button->setEnabled(false);
            no_button->setEnabled(false);
        }
    }

    void MainPanel::publish_time_handler(const QString& command_text) {
        // publish a message about when to publish the selected points
        std_msgs::Bool msg;
        if(command_text.toStdString() == "Yes") {
            msg.data = true;
        }
        else {
            msg.data = false;
        }
        publish_time_pub.publish(msg);
    }

    void MainPanel::extract_type_handler(const QString& command_text) {
        // publish a message about interaction type either weed or plant
        std_msgs::String msg;
        msg.data = command_text.toStdString();
        mode_pub.publish(msg);
    }

    void MainPanel::yes_button_handler() {
        std_msgs::Bool msg;
        msg.data = true;
        verification_pub.publish(msg);
        verification_label->setText("");
        yes_button->setEnabled(false);
        no_button->setEnabled(false);
    }

    void MainPanel::no_button_handler() {
        std_msgs::Bool msg;
        msg.data = false;
        verification_pub.publish(msg);
        verification_label->setText("");
        yes_button->setEnabled(false);
        no_button->setEnabled(false);
    }

    void MainPanel::hide_gripper_handler() {
        std_msgs::Bool msg;
        // The value doesn't matter, we just want to notify the python script to hide the gripper
        msg.data = true;
        hide_gripper_pub.publish(msg);
    }
} // namespace rviz_custom_panel

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_custom_panel::MainPanel, rviz::Panel)