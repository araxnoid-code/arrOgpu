use pollster::FutureExt;
use wgpu::{
    Adapter, Backends, Device, ExperimentalFeatures, Features, Instance, InstanceDescriptor,
    Limits, Queue, RequestAdapterOptions, wgt::DeviceDescriptor,
};

use crate::{ArrOgpuModuleInit, WgpuInit};

pub struct WgpuModule {
    pub device: Device,
    pub queue: Queue,
    pub adapter: Adapter,
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

        let (adapter, device, queue) = match arr_o_gpu_init.wgpu {
            WgpuInit::ManualDeviceQueue(adapter, device, queue) => (adapter, device, queue),
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
                            experimental_features: ExperimentalFeatures::disabled(),
                            required_limits: Limits {
                                max_storage_buffer_binding_size: arr_o_gpu_init
                                    .limits
                                    .max_storage_buffer_binding_size,

                                max_buffer_size: arr_o_gpu_init.limits.max_buffer_size,

                                max_compute_invocations_per_workgroup: arr_o_gpu_init
                                    .limits
                                    .max_compute_invocations_per_workgroup,

                                max_compute_workgroup_size_x: arr_o_gpu_init
                                    .limits
                                    .max_compute_workgroup_size_x,

                                max_compute_workgroup_size_y: arr_o_gpu_init
                                    .limits
                                    .max_compute_workgroup_size_y,

                                max_compute_workgroup_size_z: arr_o_gpu_init
                                    .limits
                                    .max_compute_workgroup_size_z,

                                max_compute_workgroups_per_dimension: arr_o_gpu_init
                                    .limits
                                    .max_compute_workgroups_per_dimension,
                                ..Default::default()
                            },
                            trace: wgpu::Trace::Off,
                        }),
                    )
                    .block_on()
                    .unwrap();

                (adapter, device, queue)
            }
        };

        Self {
            device,
            queue,
            adapter,
        }
    }

    // create array
}
