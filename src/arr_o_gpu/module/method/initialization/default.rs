use crate::{ ArrOgpuModule, ArrOgpuModuleInit };

impl Default for ArrOgpuModule {
    fn default() -> Self {
        Self::init(ArrOgpuModuleInit::default()).unwrap()
    }
}
