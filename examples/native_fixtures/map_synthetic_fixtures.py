"""Map every synthetic material fixture with provisional optical values."""

import numpy as np

from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def main() -> None:
    config = phase0_provisional_mapping()
    print(f"configuration_digest={config.digest}")
    for fixture in all_synthetic_fixtures():
        optical = map_material_volume_to_optical(fixture.volume, config)
        sigma_a_range = (
            float(np.min(optical.sigma_a)),
            float(np.max(optical.sigma_a)),
        )
        sigma_s_range = (
            float(np.min(optical.sigma_s)),
            float(np.max(optical.sigma_s)),
        )
        print(
            f"{fixture.manifest.name}: "
            f"sigma_a_range={sigma_a_range}, sigma_s_range={sigma_s_range}"
        )


if __name__ == "__main__":
    main()
