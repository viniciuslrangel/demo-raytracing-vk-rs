use vulkano::padded::Padded;
use crate::app::shader;

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub(in super) index: usize,
    pub(in super) dirty: bool,

    pub color: [f32; 3],
    pub emission: [f32; 3],
    pub smoothness: f32,
}

impl Material {
    pub fn new() -> Self {
        Self {
            index: usize::MAX,
            dirty: true,
            color: [1.0, 1.0, 1.0],
            emission: [0.0, 0.0, 0.0],
            smoothness: 0.5,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn color(&mut self, color: [f32; 3]) -> &mut Self {
        self.color = color;
        self
    }

    pub fn emission(&mut self, emission: [f32; 3]) -> &mut Self {
        self.emission = emission;
        self
    }

    pub fn smoothness(&mut self, smoothness: f32) -> &mut Self {
        self.smoothness = smoothness;
        self
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}

impl Into<shader::raytrace::fs::Material> for Material {
    fn into(self) -> shader::raytrace::fs::Material {
        shader::raytrace::fs::Material {
            color: self.color.into(),
            emission: self.emission.into(),
            smoothness: self.smoothness.into(),
        }
    }
}

impl<const N: usize> Into<::vulkano::padded::Padded<shader::raytrace::fs::Material, N>> for Material {
    fn into(self) -> Padded<shader::raytrace::fs::Material, N> {
        Padded(self.into())
    }
}