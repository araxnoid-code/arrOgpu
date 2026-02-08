use crate::AddOpt;

#[derive(Clone)]
pub struct FunctionExecuteOpt {
    pub add: AddOpt,
}

impl Default for FunctionExecuteOpt {
    fn default() -> Self {
        Self {
            add: Default::default(),
        }
    }
}
