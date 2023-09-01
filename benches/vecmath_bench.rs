use criterion::{black_box, Criterion, criterion_group, criterion_main};

use vecmath::*;

pub fn vector3_dot_vector3(c: &mut Criterion) {
    let v1 = black_box(Vector3f::new(1.5, 2.0, -1.0));
    let v2 = Vector3f::new(3.45, -8.36, 6.48);

    c.bench_function("vector3 dot vector3", |b| b.iter(|| dot(v1, v2)));
}

pub fn vector3_add_vector3(c: &mut Criterion) {
    let v1 = black_box(Vector3f::new(1.5, 2.0, -1.0));
    let v2 = Vector3f::new(3.45, -8.36, 6.48);

    c.bench_function("vector3 add vector3", |b| b.iter(|| v1 + v2));
}

pub fn matrix4x4_mul_vector3(c: &mut Criterion) {
    let mut v = black_box(Vector3f::new(2.0, -1.5, 3.0));
    let m = Matrix4x4f::rotate_axis(Vector3f::new(-1.5, -2.0, -3.5), 0.5);

    c.bench_function("matrix4x4 mul vector3", |b| b.iter(|| v = &m * v));
}

criterion_group!(benches, vector3_dot_vector3, vector3_add_vector3, matrix4x4_mul_vector3);
criterion_main!(benches);
