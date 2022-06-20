#include "plant_selector/main_panel.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <QColor>
#include <QSlider>
#include <QLabel>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QPushButton>
#include <QWheelEvent>

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

PLUGINLIB_EXPORT_CLASS(rviz_custom_panel::MainPanel, rviz::Panel)

namespace rviz_custom_panel
{
    /**
     * Constructor of the panel, initializes member variables and creates the UI
     */
    MainPanel::MainPanel(QWidget * parent):rviz::Panel(parent) {

        // Construct and lay out labels and slider controls.
        QPushButton* branch_button = new QPushButton("&Branch", this);
        QPushButton* weed_button = new QPushButton("&Weed", this);

        verification_label = new QLabel("Interact with object?", this);
        yes_button = new QPushButton("&Yes", this);
        no_button = new QPushButton("&No", this);

        cancel_button = new QPushButton("&Cancel", this);

        QGridLayout* controls_layout = new QGridLayout();
        controls_layout->addWidget(branch_button, 0, 0);
        controls_layout->addWidget(weed_button, 0, 1);
        controls_layout->addWidget(verification_label, 1, 0);
        controls_layout->addWidget(yes_button, 1, 1);
        controls_layout->addWidget(no_button, 1, 2);
        controls_layout->addWidget(cancel_button, 2, 2);

        // Construct and lay out render panel.
        render_panel = new rviz::RenderPanel();
        QVBoxLayout* main_layout = new QVBoxLayout;
        main_layout->addLayout(controls_layout);

        // Set the top-level layout for this widget.
        setLayout(main_layout);
        
        // Make signal/slot connections.
        connect(branch_button, &QPushButton::clicked, this, &MainPanel::branch_button_handler);
        connect(weed_button, &QPushButton::clicked, this, &MainPanel::weed_button_handler);

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

    void MainPanel::branch_button_handler() {
        return;
    }

    void MainPanel::weed_button_handler() {
        return;
    }
} // namespace rviz_custom_panel
