use crate::{AddOpt, DivOpt, MatmulOpt, MulOpt, SubOpt};

#[derive(Clone)]
pub struct FunctionExecuteOpt {
    pub add: AddOpt,
    pub sub: SubOpt,
    pub mul: MulOpt,
    pub div: DivOpt,
    pub matmul: MatmulOpt,
}

impl Default for FunctionExecuteOpt {
    fn default() -> Self {
        Self {
            add: Default::default(),
            sub: Default::default(),
            div: Default::default(),
            mul: Default::default(),
            matmul: Default::default(),
        }
    }
}
