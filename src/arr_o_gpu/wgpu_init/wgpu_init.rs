use pollster::FutureExt;
use wgpu::{
    Backends,
    Device,
    Features,
    Instance,
    InstanceDescriptor,
    Limits,
    MemoryHints,
    PowerPreference,
    Queue,
    RequestAdapterOptions,
    wgt::DeviceDescriptor,
};

pub struct WgpuInit {
    pub device: Device,
    pub queue: Queue,
}

impl WgpuInit {
    // initialization
    pub fn init() -> WgpuInit {
        let instance = Instance::new(
            &(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            })
        );

        let adapter = instance
            .request_adapter(
                &(RequestAdapterOptions {
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    power_preference: PowerPreference::LowPower,
                })
            )
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &(DeviceDescriptor {
                    label: Some("create device and queue"),
                    memory_hints: MemoryHints::default(),
                    required_features: Features::empty(),
                    required_limits: Limits::defaults(),
                    trace: wgpu::Trace::Off,
                })
            )
            .block_on()
            .unwrap();

        Self { device, queue }
    }

    // create array
}
