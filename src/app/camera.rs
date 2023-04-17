use cgmath::{Matrix4, SquareMatrix};
use crate::app::shader;

pub struct Camera {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub blur: f32,

    pub speed: f32,

    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            blur: 0.0,
            speed: 5.0,
            view: Matrix4::identity(),
            projection: Matrix4::identity(),
        }
    }

    pub fn set_perspective(&mut self, fov: f32, aspect: f32, near: f32, far: f32) {
        self.projection = cgmath::perspective(cgmath::Deg(fov), aspect, near, far);
    }

    pub fn set_orthographic(&mut self, left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) {
        self.projection = cgmath::ortho(left, right, bottom, top, near, far);
    }

    pub fn update_view(&mut self) {
        let rotation = Matrix4::from_angle_y(cgmath::Rad(self.rotation[1]))
            * Matrix4::from_angle_x(cgmath::Rad(self.rotation[0]))
            * Matrix4::from_angle_z(cgmath::Rad(self.rotation[2]));
        let translation = Matrix4::from_translation(self.position.into());
        self.view = translation * rotation;
    }

    pub(crate) fn move_by(&mut self, mov_x: f32, mov_y: f32, mov_z: f32, delta: f32) {
        let speed = self.speed * delta;
        let mut new_pos = self.position;
        new_pos[0] -= mov_x * speed;
        new_pos[1] -= mov_y * speed;
        new_pos[2] -= mov_z * speed;
        self.position = new_pos;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Into<shader::raytrace::fs::ViewData> for &Camera {
    fn into(self) -> shader::raytrace::fs::ViewData {
        shader::raytrace::fs::ViewData {
            proj: self.projection.into(),
            worldview: self.view.into(),
            blur: self.blur.into(),
        }
    }
}
