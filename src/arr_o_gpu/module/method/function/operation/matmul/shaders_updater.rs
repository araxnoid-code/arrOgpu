pub(crate) fn workgroup_matmul_shader_updater(
    shaders: &'static str,
    workgroup_size: u32,
) -> String {
    let mut lines = shaders.split("\n").collect::<Vec<&str>>();

    let tile_a = format!(
        "var<workgroup> tile_a: array<array<f32, {}>, {}>;",
        workgroup_size, workgroup_size
    );

    let tile_b = format!(
        "var<workgroup> tile_b: array<array<f32, {}>, {}>;",
        workgroup_size, workgroup_size
    );
    lines[2] = tile_a.as_str();
    lines[3] = tile_b.as_str();

    println!("{:?}", lines);

    lines.join("\n")
}
