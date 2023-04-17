use std::cell::RefCell;
use std::rc::Rc;

use imgui::AngleSlider;

use raytracing_demo::app::app::App;

fn main() {
    let device_name: Rc<RefCell<String>> = Rc::new(RefCell::new("Unknown".to_string()));

    let device_name_inner = device_name.clone();
    let mut app = App::create(move |_run, ui, scene, info| {
        ui.window("Camera##camera")
            // .opened()
            .position([0.0, 0.0], imgui::Condition::FirstUseEver)
            .size([300.0, 550.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text("Position");
                let cam_pos = &mut scene.camera.position;
                ui.slider("X##x", -10.0, 10.0, &mut cam_pos[0]);
                ui.slider("Y##y", -10.0, 10.0, &mut cam_pos[1]);
                ui.slider("Z##z", -10.0, 10.0, &mut cam_pos[2]);

                ui.text("Rotation");
                let cam_rot = &mut scene.camera.rotation;
                AngleSlider::new("X##rx").range_degrees(-90.0, 90.0).build(&ui, &mut cam_rot[0]);
                AngleSlider::new("Y##ry").range_degrees(-360.0, 360.0).build(&ui, &mut cam_rot[1]);
                AngleSlider::new("Z##rz").range_degrees(-180.0, 180.0).build(&ui, &mut cam_rot[2]);

                ui.text("Blur");
                ui.slider("Blur##blur", 0.0, 1.0, &mut scene.camera.blur);

                ui.text("Sample count");
                ui.slider("Sample count##sample_count", 1, 512, &mut scene.sample_count);

                ui.text("View");
                if ui.radio_button_bool("Color##color", scene.current_view == 0) {
                    scene.current_view = 0;
                }
                if ui.radio_button_bool("Color - no denoiser##color-nd", scene.current_view == 1) {
                    scene.current_view = 1;
                }
                if ui.radio_button_bool("Albedo##color", scene.current_view == 2) {
                    scene.current_view = 2;
                }
                if ui.radio_button_bool("Normal##color", scene.current_view == 3) {
                    scene.current_view = 3;
                }
                if ui.radio_button_bool("Depth##color", scene.current_view == 4) {
                    scene.current_view = 4;
                }

                ui.text("Denoiser");
                ui.slider("Kernel size", 0, 10, &mut scene.kernel_size);
                ui.slider("Kernel offset", 1, 4, &mut scene.kernel_offset);
                ui.slider("Albedo weight", 0.001, 4.0, &mut scene.denoiser_albedo_weight);
                ui.slider("Normal weight", 0.001, 4.0, &mut scene.denoiser_normal_weight);
                ui.slider("Depth weight", 0.001, 4.0, &mut scene.denoiser_depth_weight);
            });
        ui.window("Materials##materials")
            .position([0.0, 550.0], imgui::Condition::FirstUseEver)
            .size([300.0, 350.0], imgui::Condition::FirstUseEver)
            .build(|| {
                scene.all_materials.iter_mut().enumerate().for_each(|(i, mat)| {
                    let _material_id = ui.push_id(i.to_string());
                    if ui.collapsing_header(format!("Material {}", i), imgui::TreeNodeFlags::BULLET) {
                        if ui.color_edit3("Color##color", &mut mat.color) {
                            mat.mark_dirty();
                        }
                        if ui.color_edit3("Emission##emission", &mut mat.emission) {
                            mat.mark_dirty();
                        }
                        if ui.slider("Smoothness##smoothness", 0.0, 1.0, &mut mat.smoothness) {
                            mat.mark_dirty();
                        }
                    }
                });
            });

        ui.window("Info##info")
            .position([900.0, 0.0], imgui::Condition::FirstUseEver)
            .size([300.0, 65.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text("FPS: ");
                ui.same_line();
                ui.text(format!("{}", info.fps));

                ui.text("Device: ");
                ui.same_line();
                ui.text(format!("{}", device_name_inner.borrow()));
            });
        ui.window("Objects##objects")
            .position([900.0, 65.0], imgui::Condition::FirstUseEver)
            .size([300.0, 500.0], imgui::Condition::FirstUseEver)
            .build(|| {
                ui.text("Circles");
                scene.all_circles.iter_mut().enumerate().for_each(|(i, circle)| {
                    let _circle_id = ui.push_id(i.to_string());
                    if ui.collapsing_header(format!("Circle {}", i), imgui::TreeNodeFlags::BULLET) {
                        ui.text("Position");
                        if ui.slider("X##cx", -20.0, 20.0, &mut circle.position[0]) {
                            circle.mark_dirty();
                        }
                        if ui.slider("Y##cy", -20.0, 20.0, &mut circle.position[1]) {
                            circle.mark_dirty();
                        }
                        if ui.slider("Z##cz", -20.0, 20.0, &mut circle.position[2]) {
                            circle.mark_dirty();
                        }
                        ui.text("Radius");
                        if ui.slider("##cr", 0.0, 10.0, &mut circle.radius) {
                            circle.mark_dirty();
                        }
                    }
                });
            });
    });

    device_name.as_ref().replace(app.vulkan.device_name.clone());

    app.add_material() // 0
        .color([1.0, 0.2, 0.2])
        .smoothness(0.1)
    // .emission([1.0, 0.0, 0.0])
    ;

    app.add_material() // 1
        .color([0.2, 1.0, 0.2])
        .smoothness(0.5)
    // .emission([0.0, 1.0, 0.0])
    ;

    app.add_material() // 2
        .color([0.2, 0.2, 1.0])
        .smoothness(0.8)
    // .emission([0.0, 0.0, 1.0])
    ;

    const INTENSITY: f32 = 0.7;
    app.add_material() // 3
        .color([0.7, 1.0, 0.03])
        .emission([1.0 * INTENSITY, 0.917 * INTENSITY, 0.564 * INTENSITY]) // 4700K
    ;

    app.add_material() // 4
        .color([0.4, 0.4, 0.4])
        .smoothness(0.84)
    ;

    app.add_circle()
        .position([1.0, 0.3, 0.3])
        .radius(0.3)
        .material(0);

    app.add_circle()
        .position([0.0, 1.3, 0.3])
        .radius(0.3)
        .material(1);

    app.add_circle()
        .position([-1.0, 0.3, 0.3])
        .radius(0.3)
        .material(2);

    app.add_circle()
        .position([-50.0, 5.0, 50.0])
        .radius(15.0)
        .material(3);

    app.add_circle()
        .position([80.0, 30.0, 0.0])
        .radius(25.0)
        .material(3);

    app.add_circle()
        .position([0.0, -100.0, 0.0])
        .radius(100.0)
        .material(4);

    app.main_loop();
}
