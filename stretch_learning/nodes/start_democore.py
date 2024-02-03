# import roslaunch

# package = 'rviz'
# executable = 'rviz'
# node = roslaunch.core.Node(package, executable)

# launch = roslaunch.scriptapi.ROSLaunch()
# launch.start()

# process = launch.launch(node)
# print(process.is_alive())
# process.stop()

# # import roslaunch 
# # import rospy 

# # cli_args = ['/home/strech/catkin_ws/src/stretch_ros/engineered_skills/launch/demo_sp23.launch', 'map_yaml:=${HELLO_FLEET_PATH}/maps/12012023.yaml'] 
# # roslaunch_args = cli_args[1:]
# # roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
# # print(roslaunch_file)
# # print(len(roslaunch_file))
# # print(roslaunch_file[0])/home/strech/catkin_ws/src/stretch_ros/stretch_learning/checkpoints
# # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
# # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file) 
# # parent.start() 

import subprocess

process = subprocess.Popen(["roslaunch", "engineered_skills", "demo_sp23.launch", 'map_yaml:=/home/strech/stretch_user/maps/12012023.yaml'])
try:
    print('Running in process', process.pid)
    # process.wait(timeout=1000)
except KeyboardInterrupt:
    print('Timed out - killing', process.pid)
    process.kill()
print("Done")


