"""Probe Mitsuba's heterogeneous-medium and dielectric IOR parameter boundary."""

import mitsuba as mi


def main() -> None:
    mi.set_variant("scalar_rgb")
    mi.load_dict({"type": "heterogeneous", "sigma_t": 2.0, "albedo": 0.5})
    try:
        mi.load_dict(
            {
                "type": "heterogeneous",
                "sigma_t": 2.0,
                "albedo": 0.5,
                "ior": {"type": "constvolume", "value": 1.5},
            }
        )
    except RuntimeError as error:
        if "unreferenced property" not in str(error):
            raise
        print("heterogeneous spatial ior: rejected as an unreferenced property")
    else:
        raise RuntimeError("Mitsuba unexpectedly accepted heterogeneous spatial IOR")

    dielectric = mi.load_dict({"type": "dielectric", "int_ior": 1.5, "ext_ior": 1.0})
    eta = float(mi.traverse(dielectric)["eta"])
    if eta != 1.5:
        raise RuntimeError(f"unexpected dielectric eta: {eta}")
    print(f"dielectric scalar int_ior/ext_ior: accepted; eta={eta}")


if __name__ == "__main__":
    main()
