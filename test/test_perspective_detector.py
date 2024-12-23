from src.model.perspective_detector import PerspectiveDetector
import torch
import matplotlib.pyplot as plt


def test_perspective_detector() -> None:
    image: torch.Tensor = torch.tensor(plt.imread("./test/test_data/data/ecg_scans/10_1.png"))
    image = image.mean(2).float()
    binary_image: torch.Tensor = (image > 0.6) & (image < 0.95)

    pd = PerspectiveDetector(num_thetas=100)
    corrected_image, source_points = pd(binary_image)

    expected_source_points = torch.tensor(
        [[2.9672, 99.4740], [2143.5662, 41.2222], [2181.7285, 1580.0446], [41.0297, 1634.2791]]
    )
    assert torch.allclose(source_points, expected_source_points, atol=10)
    assert corrected_image.shape == binary_image.shape
