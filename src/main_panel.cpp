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

        QLabel* publishing_label = new QLabel("Instantly Publish Selection?", this);

        QStringList pub_commands = {"Yes", "No"};
        QComboBox* pub_combo = new QComboBox(this);
        pub_combo->addItems(pub_commands);

        QLabel* inter_label = new QLabel("Interaction Type:", this);

        QStringList commands = {"Branch", "Weed"};
        QComboBox* combo = new QComboBox(this);
        combo->addItems(commands);

        // eventually want to hide this and only show it when appropriate
        verification_label = new QLabel("Verification", this);
        yes_button = new QPushButton("&Yes", this);
        no_button = new QPushButton("&No", this);

        cancel_button = new QPushButton("&Cancel", this);

        QGridLayout* controls_layout = new QGridLayout();
        controls_layout->addWidget(publishing_label, 0, 0);
        controls_layout->addWidget(pub_combo, 0, 1);
        controls_layout->addWidget(inter_label, 1, 0);
        controls_layout->addWidget(combo, 1, 1);
        controls_layout->addWidget(cancel_button, 1, 2);
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
        connect(pub_combo, &QComboBox::currentTextChanged, this, &MainPanel::publish_time_changed);
        connect(combo, &QComboBox::currentTextChanged, this, &MainPanel::command_changed);
        connect(cancel_button, &QPushButton::clicked, this, &MainPanel::cancel_button_handler);

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

    void MainPanel::publish_time_changed(const QString& command_text) {
        // publish a message about interaction type
        std_msgs::Bool msg;
        if(command_text.toStdString() == "Yes") {
            msg.data = true;
        }
        else {
            msg.data = false;
        }
        publish_time_pub.publish(msg);
    }

    void MainPanel::command_changed(const QString& command_text) {
        // publish a message about interaction type
        std_msgs::String msg;
        msg.data = command_text.toStdString();
        mode_pub.publish(msg);
    }

    void MainPanel::cancel_button_handler() {
        // publish a message to delete anything going on
        std_msgs::String msg;
        msg.data = "Cancel";
        mode_pub.publish(msg);
    }
} // namespace rviz_custom_panel

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_custom_panel::MainPanel, rviz::Panel)