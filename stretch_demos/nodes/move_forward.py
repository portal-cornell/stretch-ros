import stretch_body.robot
import time

robot = stretch_body.robot.Robot()
robot.startup()

robot.base.set_translate_velocity(v_m=1.0)
# robot.base.translate_by(x_m=3.2)
robot.push_command()
time.sleep(11.0)
robot.base.set_translate_velocity(v_m=0.0)

robot.stop()