from vedo import Volume, Plotter
import os


def visualize_nifti(paths, bg=(1, 1, 1), mesh_color=(1, 0, 0)):
    plt = Plotter(N=len(paths))

    # Load a NIfTI file
    volumes = []
    for path in paths:
        print(path)
        vol = Volume(path)
        volumes.append(vol)

    # Show the volume
    for i in range(2):
        plt.at(i).show(volumes[i], bg=bg, axes=1)

    plt.interactive().close()


brain_data_dir = "C:/Work/Study/Thesis/unet_segmentation/Images"
images = os.listdir(brain_data_dir)
local_test_file = "test.nii"
visualize_nifti([os.path.join(brain_data_dir, images[0]), local_test_file])
