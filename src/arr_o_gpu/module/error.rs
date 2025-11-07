use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    Init(String),
    Matmul2D(String),
    MatmulND(String),
    Indexing(String),
    Add(String),
    Sub(String),
    Mul(String),
    Div(String),
    Broadcast(String),
    Poll(PollError),
}
