use wgpu::PollError;

#[derive(Debug)]
pub enum ArrOgpuErr {
    // module
    ModuleInit(String),

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

    // function
    Pow(String),

    // View
    Indexing(String),
    Broadcast(String),
    Slicing(String),
    Reshape(String),
    Permute(String),

    // tools
    NegativeIndexing(String),
    Padding(String),

    // pool
    Poll(String, PollError),

    // Refactor
    Refactor(String),
}

impl ArrOgpuErr {
    pub fn refactor_err_0_1_0_5() -> ArrOgpuErr {
        ArrOgpuErr::Refactor("Refactor Error | Updating Array MetaData".to_string())
    }
}
