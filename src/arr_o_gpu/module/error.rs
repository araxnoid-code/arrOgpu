use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    Init(String),
    ArangeInit(String),
    Matmul2D(String),
    MatmulND(String),
    Indexing(String),
    Add(String),
    Sub(String),
    Mul(String),
    Div(String),
    Broadcast(String),
    Slicing(String),
    Poll(PollError),
}
