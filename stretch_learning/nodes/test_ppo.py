import math
import matplotlib.pyplot as plt
import numpy as np

# from shapely.geometry import LineString


def project_point(angle, test_point):
    # Unit vector coordinates
    unit_vector = (math.cos(angle), math.sin(angle))

    # Dot product of the test point and unit vector
    dot_product = test_point[0] * unit_vector[0] + test_point[1] * unit_vector[1]

    # Projected point coordinates
    projected_point = (dot_product * unit_vector[0], dot_product * unit_vector[1])
    print(projected_point)
    if np.sign(projected_point[0]) == 0 or np.sign(projected_point[0]) == np.sign(
        unit_vector[0]
    ):
        return True
    else:
        return False


def is_point_in_half_circle(rotation_angle, center, radius, test_point):
    # Translate the test point coordinates relative to the center of the circle
    translated_point = [test_point[0] - center[0], test_point[1] - center[1]]

    # Calculate the projection of the translated point onto a vector defined by the rotation angle
    projection = project_point(rotation_angle, translated_point)
    print(projection)
    if projection and np.linalg.norm(translated_point) <= radius:
        return True
    else:
        return False


from matplotlib.patches import Polygon


def get_rotated_rect(center, width, height, angle):
    angle = math.radians(angle)
    w = width / 2
    h = height / 2

    x1 = center[0] + w * math.cos(angle) - h * math.sin(angle)
    y1 = center[1] + w * math.sin(angle) + h * math.cos(angle)

    x2 = center[0] - w * math.cos(angle) - h * math.sin(angle)
    y2 = center[1] - w * math.sin(angle) + h * math.cos(angle)

    x3 = center[0] - w * math.cos(angle) + h * math.sin(angle)
    y3 = center[1] - w * math.sin(angle) - h * math.cos(angle)

    x4 = center[0] + w * math.cos(angle) + h * math.sin(angle)
    y4 = center[1] + w * math.sin(angle) - h * math.cos(angle)

    # Create a polygon patch for the rotated rectangle
    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], closed=True)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add the polygon patch to the axis
    ax.add_patch(polygon)

    # Set the aspect ratio of the plot to equal
    ax.set_aspect("equal")

    # Set the limits of the plot
    ax.set_xlim(min(x1, x2, x3, x4), max(x1, x2, x3, x4))
    ax.set_ylim(min(y1, y2, y3, y4), max(y1, y2, y3, y4))

    # Save the figure to the specified file path
    plt.savefig("/share/portal/nlc62/hal-skill-repo/ppo_delta_3d/test_figs/test_rect")

    # Close the plot to release resources
    plt.close(fig)

    # Return the coordinates as a list of tuples
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


import math


def get_rotated_rect_point(coords, angle, pivot):
    angle = math.radians(angle)

    x1 = (
        math.cos(angle) * (coords[0][0] - pivot[0])
        - math.sin(angle) * (coords[0][1] - pivot[1])
        + pivot[0]
    )
    y1 = (
        math.sin(angle) * (coords[0][0] - pivot[0])
        + math.cos(angle) * (coords[0][1] - pivot[1])
        + pivot[1]
    )

    x2 = (
        math.cos(angle) * (coords[1][0] - pivot[0])
        - math.sin(angle) * (coords[1][1] - pivot[1])
        + pivot[0]
    )
    y2 = (
        math.sin(angle) * (coords[1][0] - pivot[0])
        + math.cos(angle) * (coords[1][1] - pivot[1])
        + pivot[1]
    )

    x3 = (
        math.cos(angle) * (coords[2][0] - pivot[0])
        - math.sin(angle) * (coords[2][1] - pivot[1])
        + pivot[0]
    )
    y3 = (
        math.sin(angle) * (coords[2][0] - pivot[0])
        + math.cos(angle) * (coords[2][1] - pivot[1])
        + pivot[1]
    )

    x4 = (
        math.cos(angle) * (coords[3][0] - pivot[0])
        - math.sin(angle) * (coords[3][1] - pivot[1])
        + pivot[0]
    )
    y4 = (
        math.sin(angle) * (coords[3][0] - pivot[0])
        + math.cos(angle) * (coords[3][1] - pivot[1])
        + pivot[1]
    )

    # Create a polygon patch for the rotated rectangle
    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], closed=True)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add the polygon patch to the axis
    ax.add_patch(polygon)

    # Set the aspect ratio of the plot to equal
    ax.set_aspect("equal")

    # Set the limits of the plot
    ax.set_xlim(min(x1, x2, x3, x4), max(x1, x2, x3, x4))
    ax.set_ylim(min(y1, y2, y3, y4), max(y1, y2, y3, y4))

    # Save the figure to the specified file path
    plt.savefig("/share/portal/nlc62/hal-skill-repo/ppo_delta_3d/test_figs/test_rect")

    # Close the plot to release resources
    plt.close(fig)

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


import numpy as np


def rotate_point(point, angle):
    """Rotates a point around the origin by the given angle.

    Args:
        point: The point to rotate.
        angle: The angle to rotate by, in radians.

    Returns:
        The rotated point.
    """
    x, y = point
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle)


def translate_point(point, offset):
    """Translates a point by the given offset.

    Args:
        point: The point to translate.
        offset: The offset to translate by.

    Returns:
        The translated point.
    """
    x, y = point
    dx, dy = offset
    return (x + dx, y + dy)


def rotate_odom(coord, angle, pivot):
    x1 = (
        math.cos(angle) * (coord[0] - pivot[0])
        - math.sin(angle) * (coord[1] - pivot[1])
        + pivot[0]
    )
    y1 = (
        math.sin(angle) * (coord[0] - pivot[0])
        + math.cos(angle) * (coord[1] - pivot[1])
        + pivot[1]
    )

    # Create a polygon patch for the rotated rectangle

    plt.scatter([x1], [y1])
    plt.scatter([coord[0]], [coord[1]])

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # Save the figure to the specified file path
    plt.savefig(
        "/home/strech/catkin_ws/src/stretch_ros/stretch_learning/nodes/plots/base_ppo/test_rotate"
    )

    # Close the plot to release resources
    plt.close()

    return (x1, y1)


def rotate_vector(vector, angle):
    x, y = vector

    x_rot = x * math.cos(angle) - y * math.sin(angle)
    y_rot = x * math.sin(angle) + y * math.cos(angle)

    return [x_rot, y_rot]


if __name__ == "__main__":
    # center = (0, 0)
    # width = 4
    # height = 2
    # theta = 45

    # rotated_coords = get_rotated_rect(center, width, height, theta)
    # print(rotated_coords)

    # coords = (0, 0)
    # angle = 45
    # pivot = (-0.075, 0.13)
    # rect = rotate_odom(coords, angle, pivot)
    # print(rect)

    point = (0.03, 0.17)
    angle = math.pi / 6
    pivot = (0, 0)

    translated_point = rotate_odom(point, angle, pivot)

    # point = np.array([-1, 1, 1])
    # angle = math.pi / 4
    # tx, ty = 0.03, 0.17
    # rotated_point = rotation_matrix(point, angle, tx, ty)

    plt.scatter([point[0]], [point[1]])
    plt.scatter([translated_point[0]], [translated_point[1]])
    # Calculate axis limits from rotated point
    # Calculate axis limits
    x_min = translated_point[0] - 0.5
    x_max = translated_point[0] + 0.5
    y_min = translated_point[1] - 0.5
    y_max = translated_point[1] + 0.5

    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)

    plt.show()
