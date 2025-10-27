use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    Init(String),
    Matmul2D(String),
    Indexing(String),
    Add(String),
    Sub(String),
    Poll(PollError),
}
