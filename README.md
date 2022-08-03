# Plant Selector

## Overview

## Requirements

### ZED SDK
Since this project assumes the use of a zed camera, it is neccesary to install the ZED SDK.

Follow the download instructions at https://www.stereolabs.com/docs/ros/

You are going to want to make sure you have ZED Neural Mode for optimal performance. You can
check if neural mode is working by running: 
```
ZED_Diagnostic -ais 8
```
This command should notify you of other dependencies such as TensorRT and it may even install these for you.

To check if the zed is working, try running:

### Other Packages
This project also requires other packages such as arm_robots, hdt_michigan, gazebo_ros, and other
basic packages for using the robot.

## Running
Once you have pulled this repo into catkin_ws/src/ make sure to catkin build. If you are able to 

Looking in the launch folder, you can see many launch scripts. All of these launch files have rviz files associated with them on launch.

### zed_offline.launch
This is the most basic launch file. In order for this to work as intended, you should have a rosbag file that was recorded from a zed camera. Look in the rosbag panel section for details on how to select the 
rosbag and recording a rosbag with a zed.

To Run:
```
roslaunch plant_selector zed_offline.launch
```

ADD IMAGE

### zed_online.launch
This launch file is the next most basic launch. It will open up rviz, start the zed camera, and run all the necessary scripts. Instead of needing to select a rosbag, like in zed_offline.launch, you can just
select from the live camera feed.

To Run:
```
roslaunch plant_selector zed_online.launch
```

ADD IMAGE

### val.launch
This launch file is meant to be for live demos with Val. In order for this to work, make sure Val is on, e-stop is not pressed, and everything is connected. This also assumes the urdf file of Val has the zed connected
and has the proper transform.

Once all connections are good, make sure you have the can connection by going into hdt_adroit_driver/scripts/ and running:
```
roscd hdt_adriot_driver/scripts
sudo ./peak_usb
```
This command should connect you to Val, run:
```
ifconfig
```
If you see a can connection, you should almost be good to run Val.

One final check. Go into the roslaunch file val_off_husky.launch in arm_robots and make sure that the path for gazebo is correct. The value defaults to "/home/armlab" but should be whatever parent directory .gazebo is in for you.

```
roscd arm_robots/launch
vi val_off_husky.launch
# Check the <env name="HOME"> Line
```

Once all of the steps above are done, you should be good to run Val. Check the documentation repo in ARMLAB for help debugging. To Run:
```
roslaunch plant_selector val.launch
```

ADD IMAGE

Talk about visual servoing/repeatability

### victor_sim.launch
This launch file is meant to simulate Victor in an offline setting. This simulation could easily be moved to real-time zed data, but the point of using Victor was to test simulation for Val. So note, this launch file may be buggy.

Since it is a simulation and you don't need to set up the hardware connections, the Victor simulation should be as easier as running the following command:
```
roslaunch plant_selector victor_sim.launch
```

ADD IMAGE

### weed_eval.launch
This launch file is very different than the previous ones. Running weed_eval.launch will evaluate how good a weed prediction model is. In the current case, it is running our weed centroid model. Once you run this launch file, the weed model will be evaluated on around 85 hand picked samples of weeds from the garden.

To Run:
```
roslaunch plant_selector weed_eval.launch
```

After running the launch file, the terminal should show useful metrics of the weed model as shown below:

ADD IMAGE

In rviz you should be able to see a weed sample with it's labeled stem center and the prediction from the model. Press enter in the terminal to toggle through different weed samples.

ADD IMAGE

To configure this code to work on a different weed prediction model, go into the eval.py file and change the function that is passed into the WeedMetrics object. In order for this to work as easily as possible, your prediction model should return an x y z numpy array of where it thinks the weed stem is. Also, make sure to return the normal of the plane of the dirt. While this isn't extremely necessary, it really helps the visualization when toggling through weeds as eval.py will rotate the pointcloud to make sure all weeds are oriented in the same way.

## Code Explanation

### Weed Extraction

### Branch Extraction

## Rviz Plugins
In order for the UI aspect of this code to work, there were many necessary custom rviz plugins that needed to be created.

### Publishing Selector
The Publishing Selector plugin is an rviz tool. This means to add it, you must press the '+' icon at the top of rviz.

Publishing Selector functions almost exactly as the 'Select' Tool. The only difference is that the selected region of the tool is publish to the /rviz_selected_points topic. The original 'Select' tool does not publish to ros, making it pretty useless.

Using the publishing selector is pretty simple, below are important key shortcuts.

ADD SHORTCUT INFO

ADD IMAGE

### MainPanel

### RosbagPanel

## Miscellaneous