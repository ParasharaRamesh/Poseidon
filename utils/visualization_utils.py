import matplotlib.pyplot as plt

from utils.graph_utils import HUMAN_POSE_EDGES

def visualize_2d_pose(pose):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.scatter(pose[:, 0], pose[:, 1])

    for i, xy in enumerate(pose):
        ax.annotate(f'{i}', xy=xy)

    for joints in HUMAN_POSE_EDGES:
        x1, y1 = pose[joints[0]]
        x2, y2 = pose[joints[1]]
        ax.plot([x1, x2], [y1, y2], 'bo-')

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('2D Human Pose Skeleton')

    plt.show()

def visualize_3d_pose(data, elev=15., azim=0):
    # Create a 3D plot for the human pose
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points as markers
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    ax.scatter(x, y, z, color='b', marker='o', label='Keypoints')

    # Draw lines for each edge in HUMAN_POSE_EDGES
    for edge in HUMAN_POSE_EDGES:
        start, end = edge
        ax.plot(
            [x[start], x[end]],
            [y[start], y[end]],
            [z[start], z[end]],
            color='r', linestyle='-', linewidth=2
        )

    # Labeling the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title('3D Human Pose Skeleton')

    # Show the plot
    plt.show()