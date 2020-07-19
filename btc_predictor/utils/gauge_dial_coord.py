############################################################################
# Convenience script to calculate coordinate for the momentum gauge plot
# for the dashboard.
############################################################################
import math


def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = 0.24, 0.5
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


coords_50 = [(0.235, 0.5), (0.24, 0.65), (0.245, 0.5)]

coords_40 = []
for c in coords_50:
    coords_40 += [rotate(c, math.radians(18))]
print(f'40 degrees coordiantes:\n {coords_40}')

coords_30 = []
for c in coords_40:
    coords_30 += [rotate(c, math.radians(18))]
print(f'30 degrees coordiantes:\n {coords_30}')

coords_20 = []
for c in coords_30:
    coords_20 += [rotate(c, math.radians(18))]
print(f'20 degrees coordiantes:\n {coords_20}')

coords_10 = []
for c in coords_20:
    coords_10 += [rotate(c, math.radians(18))]
print(f'10 degrees coordiantes:\n {coords_10}')

coords_0 = []
for c in coords_10:
    coords_0 += [rotate(c, math.radians(18))]
print(f'0 degrees coordiantes:\n {coords_0}')

coords_60 = []
for c in coords_50:
    coords_60 += [rotate(c, math.radians(-18))]
print(f'60 degrees coordiantes:\n {coords_60}')

coords_70 = []
for c in coords_60:
    coords_70 += [rotate(c, math.radians(-18))]
print(f'70 degrees coordiantes:\n {coords_70}')

coords_80 = []
for c in coords_70:
    coords_80 += [rotate(c, math.radians(-18))]
print(f'80 degrees coordiantes:\n {coords_80}')

coords_90 = []
for c in coords_80:
    coords_90 += [rotate(c, math.radians(-18))]
print(f'90 degrees coordiantes:\n {coords_90}')

coords_100 = []
for c in coords_90:
    coords_100 += [rotate(c, math.radians(-18))]
print(f'100 degrees coordiantes:\n {coords_100}')
