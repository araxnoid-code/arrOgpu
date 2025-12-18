use arr_o_gpu::{ ArrOgpuErr, ArrOgpuModule, r };

fn main() -> Result<(), ArrOgpuErr> {
    let module = ArrOgpuModule::default();

    let data = (0..6).map(|v| v as f32).collect::<Vec<f32>>();
    let array = module.array_from_vector(&data, &[2, 1, 3])?;

    let array = module.permute(&array, &[2, 1, 0])?;
    let array = module.broadcast(&array, &[3, 3, 2])?;
    let array = module.slicing(&array, &[r(..), r(1..), r(..1)])?;
    let out_a = module.sum_axis(&array, &[0])?;
    println!("===========");
    println!("{}", array.contiguous());
    println!("{}", out_a);

    Ok(())
}
