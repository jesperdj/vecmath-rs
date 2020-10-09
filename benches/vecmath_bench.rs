use criterion::{black_box, Criterion, criterion_group, criterion_main};

use vecmath::*;

pub fn matrix4x4_mul_vector3(c: &mut Criterion) {
    let mut v = black_box(Vector3::new(2.0, -1.5, 3.0));
    let m = Matrix4x4::rotate_axis(Vector3::new(-1.5, -2.0, -3.5), 0.5);

    c.bench_function("matrix4x4 mul vector3", |b| b.iter(|| { v = &m * v; }));
}

criterion_group!(benches, matrix4x4_mul_vector3);
criterion_main!(benches);
