from collections import namedtuple

Material = namedtuple("Material", ["name", "thermal_cond"])

Test = namedtuple(
        "Test",
        ["test_no", "bnd", "material", "cells", "limits"],
        defaults=[[0.0, 1.0]]
        )

