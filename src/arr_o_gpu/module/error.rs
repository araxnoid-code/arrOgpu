#[derive(Debug)]
pub enum ArrOgpuErr {
    Init(String),
    Matmul2D(String),
    Indexing(String),
}
