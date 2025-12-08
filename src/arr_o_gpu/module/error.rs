use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    Init(String),
    ArangeInit(String),
    Matmul2D(String),
    MatmulND(String),
    Add(String),
    Sub(String),
    Mul(String),
    Div(String),
    DotProduct(String),
    // mutate
    Indexing(String),
    Broadcast(String),
    Slicing(String),
    Reshape(String),
    Permute(String),
    // pool
    Poll(PollError),
}
