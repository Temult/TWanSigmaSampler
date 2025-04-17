# TWanVideoSigmaSampler: EXPERIMENTAL

**THIS NODE IS CURRENTLY A PROTOTYPE AND UNDER ACTIVE DEVELOPMENT. IT IS NOT READY FOR GENERAL USE. USE AT YOUR OWN RISK!**

## Description

TWanVideoSigmaSampler is a ComfyUI custom node intended as a modified version of the WanVideoSampler, allowing for the input of custom sigma (noise level) schedules. The goal is to enable users to bypass the built-in samplers and exert fine-grained control over the diffusion process in Wan-based video generation.

## Intended Functionality

The primary purpose of this node is to allow ComfyUI workflows to leverage custom sigma schedules, potentially unlocking:

*   **Unique Visual Effects:** Fine-grained control over noise levels for specialized motion and detail.
*   **Experimental Sampling Methods:** Support for advanced sampling techniques that rely on specific, non-standard sigma curves.
*   **Precise Control:** Override default sampling behaviors for customized artistic expression.

This node is being developed to function as a (mostly) drop-in replacement for the original `WanVideoSampler`. The goal is for most optional features in the original sampler to work seamlessly with custom sigma schedules.

## Current Status

*   **Testing:** Node appears to function and is currently being tested.  

## Dependencies

*   [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (The original WanVideo custom node suite)
    * Follow its installation instructions. Place this node in the `ComfyUI/custom_nodes` directory.
*   [diffusers](https://github.com/huggingface/diffusers)
*   [accelerate](https://github.com/huggingface/accelerate)
*   [ftfy](https://pypi.org/project/ftfy/)
    *Install these dependencies using ComfyUI's `pip` inside "ComfyUI/python_embeded/":```bash
    ./python_embeded/python.exe -m pip install diffusers accelerate ftfy
    ```

## Related Projects

This node is intended to be used with custom sigma schedule generators, such as:

*   **[TWanSigmaGraph](https://github.com/Temult/TWanSigmaGraph):** A ComfyUI custom node for visually editing and generating custom sigma schedules as a graph.
