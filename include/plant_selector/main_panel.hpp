#ifndef rviz_panel_H_
#define rviz_panel_H_

#include <ros/ros.h>
#include <rviz/panel.h>

#include <string.h>
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <std_msgs/Bool.h>

namespace rviz
{
class Display;
class RenderPanel;
class VisualizationManager;
}


namespace rviz_custom_panel
{
    /**
     *  Here we declare our new subclass of rviz::Panel. Every panel which
     *  can be added via the Panels/Add_New_Panel menu is a subclass of
     *  rviz::Panel.
     */

    class MainPanel : public rviz::Panel
    {
        /**
         * This class uses Qt slots and is a subclass of QObject, so it needs
         * the Q_OBJECT macro.
         */
        Q_OBJECT

        public:
            /**
             *  QWidget subclass constructors usually take a parent widget
             *  parameter (which usually defaults to 0).  At the same time,
             *  pluginlib::ClassLoader creates instances by calling the default
             *  constructor (with no arguments). Taking the parameter and giving
             *  a default of 0 lets the default constructor work and also lets
             *  someone using the class for something else to pass in a parent
             *  widget as they normally would with Qt.
             */
            MainPanel(QWidget * parent = 0);

            /**
             *  Now we declare overrides of rviz::Panel functions for saving and
             *  loading data from the config file.  Here the data is the topic name.
             */
            virtual void save(rviz::Config config) const;
            virtual void load(const rviz::Config & config);

        /**
         *  Here we declare some internal slots.
         */
        private Q_SLOTS:

            void verification_callback(const std_msgs::Bool::ConstPtr& msg);
            void publish_time_handler(const QString& command_text);
            void extract_type_handler(const QString& command_text);
            void yes_button_handler();
            void no_button_handler();
            void hide_gripper_handler();

        /**
         *  Finally, we close up with protected member variables
         */
        protected:
            ros::NodeHandle n;
            ros::Publisher mode_pub;
            ros::Publisher publish_time_pub;
            ros::Publisher verification_pub;
            ros::Publisher hide_gripper_pub;
            ros::Subscriber ask_verification_sub;

            rviz::VisualizationManager* manager;
            rviz::RenderPanel* render_panel;

            QLabel* verification_label;
            QPushButton* yes_button;
            QPushButton* no_button;
    };
} // namespace rviz_custom_panel

#endif