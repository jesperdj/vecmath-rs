# Vector math for 2D and 3D graphics applications

Main structs: `Scalar`, `Point2`, `Vector2`, `Ray2`, `BoundingBox2` and `Point3`, `Vector3`, `Ray3`, `BoundingBox3`.

Structs for transforms: `Transform2` and `Transform3`.

A bunch of traits to support these:
- `Distance` and `RelativeDistance` (implemented for `Point2` and `Point3`).
- `Length` and `RelativeLength` (implemented for `Vector2` and `Vector3`).
- `DotProduct` (implemented for `Vector2` and `Vector3`).
- `CrossProduct` (implemented for `Vector3`).
- `Intersection` and `Union` (implemented for `BoundingBox2` and `BoundingBox3`).
- `MinMax` for things that have `min()` and `max()` methods.
- `Transform` for things that can be transformed with a `Transform2` or `Transform3`.
