use crate::{AddOpt, DivOpt, MulOpt, SubOpt};

#[derive(Clone)]
pub struct FunctionExecuteOpt {
    pub add: AddOpt,
    pub sub: SubOpt,
    pub mul: MulOpt,
    pub div: DivOpt,
}

impl Default for FunctionExecuteOpt {
    fn default() -> Self {
        Self {
            add: Default::default(),
            sub: Default::default(),
            div: Default::default(),
            mul: Default::default(),
        }
    }
}
