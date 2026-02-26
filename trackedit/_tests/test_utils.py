import numpy as np
import pytest

from trackedit.utils.utils import calculate_bbox_from_mask, create_ellipsoid_mask


def test_calculate_bbox_from_mask_3d():
    """Test bounding box calculation in 3D."""
    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[40:60, 30:70, 45:55] = True

    bbox = calculate_bbox_from_mask(mask)

    assert len(bbox) == 6  # 3D bbox
    assert bbox[0] == 40  # min_z
    assert bbox[1] == 30  # min_y
    assert bbox[2] == 45  # min_x
    assert bbox[3] == 59  # max_z
    assert bbox[4] == 69  # max_y
    assert bbox[5] == 54  # max_x


def test_calculate_bbox_from_mask_2d():
    """Test bounding box calculation in 2D."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 30:70] = True

    bbox = calculate_bbox_from_mask(mask)

    assert len(bbox) == 4  # 2D bbox
    assert bbox[0] == 40  # min_y
    assert bbox[1] == 30  # min_x
    assert bbox[2] == 59  # max_y
    assert bbox[3] == 69  # max_x


def test_calculate_bbox_from_mask_single_pixel():
    """Test bounding box for single pixel mask."""
    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[50, 50, 50] = True

    bbox = calculate_bbox_from_mask(mask)

    assert len(bbox) == 6
    assert bbox[0] == 50  # min_z
    assert bbox[1] == 50  # min_y
    assert bbox[2] == 50  # min_x
    assert bbox[3] == 50  # max_z
    assert bbox[4] == 50  # max_y
    assert bbox[5] == 50  # max_x


def test_calculate_bbox_from_empty_mask():
    """Test that empty mask raises ValueError."""
    mask = np.zeros((100, 100, 100), dtype=bool)

    with pytest.raises(ValueError, match="Mask is empty"):
        calculate_bbox_from_mask(mask)


def test_create_ellipsoid_mask_3d():
    """Test ellipsoid mask creation in 3D with anisotropic radii."""
    center = (50, 50, 50)
    radii = (5, 10, 10)  # Compressed in z dimension
    shape = (100, 100, 100)

    mask = create_ellipsoid_mask(center, radii, shape)

    assert mask.dtype == bool
    assert mask.shape == shape
    assert mask[50, 50, 50]  # Center should be True

    # Check approximate volume (ellipsoid volume = 4/3 * pi * r1 * r2 * r3)
    volume = mask.sum()
    expected_volume = (4 / 3) * np.pi * radii[0] * radii[1] * radii[2]
    assert abs(volume - expected_volume) < expected_volume * 0.2  # Within 20%


def test_create_ellipsoid_mask_2d():
    """Test ellipsoid mask creation in 2D with anisotropic radii."""
    center = (50, 50)
    radii = (10, 20)  # Different radii in y and x
    shape = (100, 100)

    mask = create_ellipsoid_mask(center, radii, shape)

    assert mask.dtype == bool
    assert mask.shape == shape
    assert mask[50, 50]  # Center should be True

    # Check approximate area (ellipse area = pi * r1 * r2)
    area = mask.sum()
    expected_area = np.pi * radii[0] * radii[1]
    assert abs(area - expected_area) < expected_area * 0.2  # Within 20%


def test_create_ellipsoid_mask_anisotropic_scaling():
    """Test that ellipsoid with scaled radii compensates for viewer scaling.

    This simulates the case where data has anisotropic scaling like (4, 1, 1).
    An ellipsoid with radii (5, 20, 20) in database space should appear
    spherical when scaled by (4, 1, 1) in viewer.
    """
    center = (25, 50, 50)
    scale = (4.0, 1.0, 1.0)
    viewer_radius = 20  # Desired radius in viewer space

    # Calculate database radii (compressed in z)
    radii = (
        viewer_radius / scale[0],  # z: 20/4 = 5
        viewer_radius / scale[1],  # y: 20/1 = 20
        viewer_radius / scale[2],  # x: 20/1 = 20
    )

    shape = (50, 100, 100)
    mask = create_ellipsoid_mask(center, radii, shape)

    # The mask should be compressed in z dimension
    bbox = calculate_bbox_from_mask(mask)

    # Check bbox sizes approximately match radii
    z_size = bbox[3] - bbox[0]
    y_size = bbox[4] - bbox[1]
    x_size = bbox[5] - bbox[2]

    # Z should be compressed (2 * radius_z ≈ 10)
    assert abs(z_size - 2 * radii[0]) <= 2
    # Y and X should be normal (2 * radius ≈ 40)
    assert abs(y_size - 2 * radii[1]) <= 2
    assert abs(x_size - 2 * radii[2]) <= 2


def test_create_ellipsoid_mask_dimension_mismatch():
    """Test that dimension mismatch raises assertion error."""
    center = (50, 50)  # 2D center
    radii = (10, 10, 10)  # 3D radii
    shape = (100, 100)  # 2D shape

    with pytest.raises(AssertionError):
        create_ellipsoid_mask(center, radii, shape)
