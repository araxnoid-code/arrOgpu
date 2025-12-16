use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    // Init
    Init(String),
    ArangeInit(String),

    // Operation
    Matmul2D(String),
    MatmulND(String),
    Add(String),
    Sub(String),
    Mul(String),
    Div(String),
    DotProduct(String),
    SumAxis(String),

    // View
    Indexing(String),
    Broadcast(String),
    Slicing(String),
    Reshape(String),
    Permute(String),
    // pool
    Poll(PollError),
}
