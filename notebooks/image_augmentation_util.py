import numpy as np
import scipy.ndimage as ndi
import nibabel as nib

def inverse_contrast_3d(image, mask):
    mask = mask.astype(bool)

    max_intensity = np.max(image[mask])

    inversed_image = image.copy()
    inversed_image[mask] = max_intensity - image[mask]

    return inversed_image


def translate_3d_image(image, translation_range):
    translation_value = np.random.randint(-translation_range, translation_range)
    print(image.shape)
    translated_image = np.roll(image, translation_value, axis=(0, 1, 2))

    print(f"Translated for {translation_value} pixels")

    return translated_image


def rotate_3d_image(image, rotation_angle):
    rot_x, rot_y, rot_z = np.random.uniform(-rotation_angle, rotation_angle, size=3)

    rot_x_image = ndi.rotate(image, rot_x, mode='nearest', axes=(1, 2), reshape=False)
    rot_y_image = ndi.rotate(rot_x_image, rot_y, mode='nearest', axes=(0, 2), reshape=False)
    rot_z_image = ndi.rotate(rot_y_image, rot_z, mode='nearest', axes=(0, 1), reshape=False)

    print(f"Rotated {rot_x} degrees in x axis")
    print(f"Rotated {rot_y} degrees in y axis")
    print(f"Rotated {rot_z} degrees in z axis")

    return rot_z_image


def augment_3d_image(image, thresh_translation, thresh_rotation_angle):
    translated_3d_image = translate_3d_image(image, thresh_translation)
    rotated_3d_image = rotate_3d_image(translated_3d_image, thresh_rotation_angle)

    return rotated_3d_image


def get_fourier_image(image_slice):
    f_image = np.fft.fft2(image_slice)
    f_image = np.fft.fftshift(f_image)

    return f_image


def get_spatial_image_from_fourier_image(fourier_image_slice):
    s_image = np.fft.ifftshift(fourier_image_slice)
    s_image = np.fft.ifft2(s_image)

    return s_image.real

def pad_image(image, max_height, max_width):
    height_diff = max_height - image.shape[0]
    width_diff = max_width - image.shape[1]
    top_pad = height_diff // 2
    bottom_pad = height_diff - top_pad
    left_pad = width_diff // 2
    right_pad = width_diff - left_pad

    # Pad the image
    padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='edge')

    return padded_image

def augment_image(image, thresh_translation, thresh_rotation_angle, slice_number, max_height, max_width):
    augmented_3d_image_1 = augment_3d_image(image, thresh_translation, thresh_rotation_angle)
    augmented_3d_image_2 = augment_3d_image(augmented_3d_image_1, thresh_translation, thresh_rotation_angle)
    augmented_3d_image_3 = augment_3d_image(augmented_3d_image_2, thresh_translation, thresh_rotation_angle)

    original_image_slice = ndi.rotate(image[:, :, slice_number], 90)
    augmented_3d_image_1_slice = ndi.rotate(augmented_3d_image_1[:, :, slice_number], 90)
    augmented_3d_image_2_slice = ndi.rotate(augmented_3d_image_2[:, :, slice_number], 90)
    augmented_3d_image_3_slice = ndi.rotate(augmented_3d_image_3[:, :, slice_number], 90)

    original_image_slice = pad_image(original_image_slice, max_height, max_width)
    augmented_3d_image_1_slice = pad_image(augmented_3d_image_1_slice, max_height, max_width)
    augmented_3d_image_2_slice = pad_image(augmented_3d_image_2_slice, max_height, max_width)
    augmented_3d_image_3_slice = pad_image(augmented_3d_image_3_slice, max_height, max_width)

    f_original_image_slice = get_fourier_image(original_image_slice)
    f_augmented_3d_image_1_slice = get_fourier_image(augmented_3d_image_1_slice)
    f_augmented_3d_image_2_slice = get_fourier_image(augmented_3d_image_2_slice)
    f_augmented_3d_image_3_slice = get_fourier_image(augmented_3d_image_3_slice)

    f_aggregated_scan = f_original_image_slice.copy()
    f_aggregated_scan[50:90, :] = f_augmented_3d_image_1_slice[50:90, :]
    f_aggregated_scan[160:185, :] = f_augmented_3d_image_2_slice[160:185, :]
    f_aggregated_scan[220: 245, :] = f_augmented_3d_image_3_slice[220:245, :]

    aggregated_scan_image = get_spatial_image_from_fourier_image(f_aggregated_scan)

    return aggregated_scan_image


def motion_corrupt_3d(image, thresh_translation, thresh_rotation_angle):
    image_array = image.get_fdata()

    augmented_3d_image_1 = augment_3d_image(image_array, thresh_translation, thresh_rotation_angle)
    augmented_3d_image_2 = augment_3d_image(augmented_3d_image_1, thresh_translation, thresh_rotation_angle)
    augmented_3d_image_3 = augment_3d_image(augmented_3d_image_2, thresh_translation, thresh_rotation_angle)

    augmented_3d_image = np.zeros_like(image_array)

    for slice in range(image_array.shape[2]):
        f_original_image_slice = get_fourier_image(image_array[:, :, slice])
        f_augmented_3d_image_1_slice = get_fourier_image(augmented_3d_image_1[:, :, slice])
        f_augmented_3d_image_2_slice = get_fourier_image(augmented_3d_image_2[:, :, slice])
        f_augmented_3d_image_3_slice = get_fourier_image(augmented_3d_image_3[:, :, slice])

        f_aggregated_scan = f_original_image_slice.copy()
        f_aggregated_scan[50:90, :] = f_augmented_3d_image_1_slice[50:90, :]
        f_aggregated_scan[160:185, :] = f_augmented_3d_image_2_slice[160:185, :]
        f_aggregated_scan[220: 245, :] = f_augmented_3d_image_3_slice[220:245, :]

        aggregated_scan_image = get_spatial_image_from_fourier_image(f_aggregated_scan)

        augmented_3d_image[:, :, slice] = aggregated_scan_image

    resultant_image = nib.Nifti1Image(augmented_3d_image, affine=image.affine)

    return resultant_image