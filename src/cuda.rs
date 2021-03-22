// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            module: module,
            stream: stream,
            _context: _context
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut device_input = DeviceBox::new(input)?;
        let mut device_conv2d_output = DeviceBox::new(&ConvOutput{0: [[[0f64; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]})?;

        const NUM_BLOCKS_PER_OUTPUT_NEURON: u32 = 8;
        let mut device_block_outputs = DeviceBuffer::from_slice(&[[0f64; NUM_BLOCKS_PER_OUTPUT_NEURON as usize]; OUT_LAYER_SIZE])?;

        let module = &self.module;
        let stream = &self.stream;

        unsafe {
            launch!(module.conv2d_relu<<<1024, 32, 0, stream>>>(
                device_input.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                device_conv2d_output.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?; 

        unsafe {
            launch!(module.flattened_to_dense<<<(NUM_BLOCKS_PER_OUTPUT_NEURON, OUT_LAYER_SIZE as u32), 256, 0, stream>>>(
                device_conv2d_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                device_block_outputs.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;

        let mut block_outputs = [[0f64; NUM_BLOCKS_PER_OUTPUT_NEURON as usize]; OUT_LAYER_SIZE];
        device_block_outputs.copy_to(&mut block_outputs)?;

        let mut output = OutputVec{0: [0f64; OUT_LAYER_SIZE]};
        for i in 0..OUT_LAYER_SIZE {
            let i = i as usize;
            for j in 0..NUM_BLOCKS_PER_OUTPUT_NEURON {
                let j = j as usize;
                output.0[i] += block_outputs[i][j];
            }
        }
        Ok(output)
    }
}
