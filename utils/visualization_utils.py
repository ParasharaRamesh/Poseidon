import numpy as np
import matplotlib.pyplot as plt
import itertools

from utils.graph_utils import HUMAN_POSE_EDGES, HUMAN_POSE_EDGES_CUSTOM

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

def visualize_2d_pose_custom(pose):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    ax.scatter(pose[:, 0], pose[:, 1])

    for i, xy in enumerate(pose):
        ax.annotate(f'{i}', xy=xy)

    for joints in HUMAN_POSE_EDGES_CUSTOM:
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

def visualize_3d_pose_custom(data, elev=15., azim=0):
    # Create a 3D plot for the human pose
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points as markers
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    ax.scatter(x, y, z, color='b', marker='o', label='Keypoints')

    # Draw lines for each edge in HUMAN_POSE_EDGES
    for edge in HUMAN_POSE_EDGES_CUSTOM:
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

def visualize_2d_pose_actions(poses, action, k=5):
    plt.figure(figsize=(20, 4))

    for i in range(k):
        pose_idx = np.random.choice(len(poses))
        pose = poses[pose_idx]

        plt.subplot(1, k, i+1)
        plt.scatter(pose[:, 0], pose[:, 1])

        for i, xy in enumerate(pose):
            plt.annotate(f'{i}', xy=xy)

        for joints in HUMAN_POSE_EDGES:
            x1, y1 = pose[joints[0]]
            x2, y2 = pose[joints[1]]
            plt.plot([x1, x2], [y1, y2], 'bo-')

        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title(f'{action} {i}')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()


    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    """
    plt.figure(figsize=(15,8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cmap='Reds'
        print("Normalized Confusion Matrix")
    else:
        cmap='Greens'
        print('Confusion Matrix Without Normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
