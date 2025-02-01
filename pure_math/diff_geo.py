import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def convert_to_list_of_vectors(arr):
    """
    Convert input to a list of 3D numpy arrays.

    Parameters
    ----------
    arr : numpy.ndarray or list
        Either a single 3D vector (numpy array of shape (3,) or list of 3 numbers)
        or a collection of such vectors (2D numpy array of shape (n,3) or list of vectors).

    Returns
    -------
    list of numpy.ndarray
        Each element is a 3D vector of shape (3,).

    Raises
    ------
    ValueError
        If the input is not properly structured.
    """
    if isinstance(arr, np.ndarray):
        if arr.ndim == 1 and arr.shape[0] == 3:
            return [arr]
        elif arr.ndim == 2 and arr.shape[1] == 3:
            return [arr[i, :] for i in range(arr.shape[0])]
        else:
            raise ValueError(
                "Array must be a 3-element vector or an array of shape (n,3)."
            )
    elif isinstance(arr, list):
        if not arr:
            raise ValueError("Input list is empty.")
        if isinstance(arr[0], (int, float, np.number)):
            arr_np = np.array(arr)
            if arr_np.shape != (3,):
                raise ValueError("A single vector must have exactly 3 elements.")
            return [arr_np]
        else:
            vecs = []
            for item in arr:
                if isinstance(item, np.ndarray):
                    if item.shape != (3,):
                        raise ValueError("Each vector must be shape (3,).")
                    vecs.append(item)
                elif isinstance(item, list):
                    item_np = np.array(item)
                    if item_np.shape != (3,):
                        raise ValueError("Each vector must have 3 elements.")
                    vecs.append(item_np)
                else:
                    raise ValueError(
                        "Elements must be numpy arrays or lists of 3 numbers."
                    )
            return vecs
    else:
        raise ValueError("Input must be a numpy array or a list.")


def plot_arrows(p, v):
    """
    Plot 3D arrows from points in p to endpoints at p + v.

    Parameters
    ----------
    p : numpy.ndarray or list
        A single 3D vector or a list/array of 3D vectors (each of shape (3,))
        representing starting points.
    v : numpy.ndarray or list
        A single 3D vector or a list/array of 3D vectors (each of shape (3,))
        representing arrow directions.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object containing the 3D plot.

    Raises
    ------
    ValueError
        If both p and v contain multiple vectors and their lengths differ.
    """
    p_list = convert_to_list_of_vectors(p)
    v_list = convert_to_list_of_vectors(v)

    if len(p_list) != len(v_list):
        if not (len(p_list) == 1 or len(v_list) == 1):
            raise ValueError(
                "Multiple vectors in both p and v must have equal lengths."
            )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Arrows Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot arrows with broadcasting rules.
    if len(p_list) == 1 and len(v_list) > 1:
        base = p_list[0]
        for arrow in v_list:
            ax.quiver(
                base[0],
                base[1],
                base[2],
                arrow[0],
                arrow[1],
                arrow[2],
                arrow_length_ratio=0.1,
                color="blue",
            )
            ax.scatter(*base, color="red")
            ax.scatter(*(base + arrow), color="green")
    elif len(v_list) == 1 and len(p_list) > 1:
        arrow = v_list[0]
        for base in p_list:
            ax.quiver(
                base[0],
                base[1],
                base[2],
                arrow[0],
                arrow[1],
                arrow[2],
                arrow_length_ratio=0.1,
                color="blue",
            )
            ax.scatter(*base, color="red")
            ax.scatter(*(base + arrow), color="green")
    else:
        for base, arrow in zip(p_list, v_list):
            ax.quiver(
                base[0],
                base[1],
                base[2],
                arrow[0],
                arrow[1],
                arrow[2],
                arrow_length_ratio=0.1,
                color="blue",
            )
            ax.scatter(*base, color="red")
            ax.scatter(*(base + arrow), color="green")

    # Compute plot limits from data; ensure 0 is included for axis clarity.
    all_pts = []
    if len(p_list) == 1 and len(v_list) > 1:
        for arrow in v_list:
            all_pts.append(p_list[0])
            all_pts.append(p_list[0] + arrow)
    elif len(v_list) == 1 and len(p_list) > 1:
        for base in p_list:
            all_pts.append(base)
            all_pts.append(base + v_list[0])
    else:
        for base, arrow in zip(p_list, v_list):
            all_pts.append(base)
            all_pts.append(base + arrow)
    all_pts = np.array(all_pts)
    margin = 1
    xmin = min(np.min(all_pts[:, 0]) - margin, 0)
    xmax = max(np.max(all_pts[:, 0]) + margin, 0)
    ymin = min(np.min(all_pts[:, 1]) - margin, 0)
    ymax = max(np.max(all_pts[:, 1]) + margin, 0)
    zmin = min(np.min(all_pts[:, 2]) - margin, 0)
    zmax = max(np.max(all_pts[:, 2]) + margin, 0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # Draw solid black coordinate axes.
    ax.plot([xmin, xmax], [0, 0], [0, 0], "k-", lw=2)
    ax.plot([0, 0], [ymin, ymax], [0, 0], "k-", lw=2)
    ax.plot([0, 0], [0, 0], [zmin, zmax], "k-", lw=2)

    # Set view angle for a clear 3D perspective.
    ax.view_init(elev=20, azim=30)

    return fig


# To save the figure as a .png:
# fig = plot_arrows(p, v)
# fig.savefig("output.png")
