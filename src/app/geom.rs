use vulkano::padded::Padded;
use crate::app::shader;

#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub(in super) index: usize,
    pub(in super) dirty: bool,

    pub position: [f32; 3],
    pub radius: f32,
    pub material: i32,
}

impl Circle {
    pub fn new() -> Self {
        Self {
            index: usize::MAX,
            dirty: true,
            position: [0.0, 0.0, 0.0],
            radius: 1.0,
            material: 0,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn position(&mut self, position: [f32; 3]) -> &mut Self {
        self.position = position;
        self
    }

    pub fn radius(&mut self, radius: f32) -> &mut Self {
        self.radius = radius;
        self
    }

    pub fn material(&mut self, material: i32) -> &mut Self {
        self.material = material;
        self
    }
}

impl Default for Circle {
    fn default() -> Self {
        Self::new()
    }
}

impl Into<shader::raytrace::fs::Circle> for Circle {
    fn into(self) -> shader::raytrace::fs::Circle {
        shader::raytrace::fs::Circle {
            position: self.position,
            radius: self.radius,
            material: self.material,
        }
    }
}

impl<const N: usize> Into<::vulkano::padded::Padded<shader::raytrace::fs::Circle, N>> for Circle {
    fn into(self) -> Padded<shader::raytrace::fs::Circle, N> {
        Padded(self.into())
    }
}
