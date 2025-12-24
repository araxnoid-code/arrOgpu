use pollster::FutureExt;
use wgpu::{
    Backends, Device, Features, Instance, InstanceDescriptor, Limits, Queue, RequestAdapterOptions,
    wgt::DeviceDescriptor,
};

use crate::{ArrOgpuModuleInit, WgpuInit};

pub struct WgpuModule {
    pub device: Device,
    pub queue: Queue,
}

impl WgpuModule {
    // initialization
    pub(crate) fn init(arr_o_gpu_init: ArrOgpuModuleInit) -> WgpuModule {
        let instance = Instance::new(
            &(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            }),
        );

        let (device, queue) = match arr_o_gpu_init.wgpu {
            WgpuInit::ManualDeviceQueue(device, queue) => (device, queue),
            WgpuInit::ManualInit(manual_init) => {
                let adapter = instance
                    .request_adapter(
                        &(RequestAdapterOptions {
                            compatible_surface: None,
                            force_fallback_adapter: false,
                            power_preference: manual_init.power.conversion(),
                        }),
                    )
                    .block_on()
                    .unwrap();

                let (device, queue) = adapter
                    .request_device(
                        &(DeviceDescriptor {
                            label: Some("create device and queue"),
                            memory_hints: manual_init.memory.conversion(),
                            required_features: Features::empty(),
                            required_limits: Limits {
                                ..Default::default()
                            },
                            trace: wgpu::Trace::Off,
                        }),
                    )
                    .block_on()
                    .unwrap();

                (device, queue)
            }
        };

        Self { device, queue }
    }

    // create array
}
