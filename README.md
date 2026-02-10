# OYMotion Hand Adaptors

Based on [roh_firmware](https://github.com/oymotion/roh_firmware)

## Installation

```bash
pip install -e .
```

## Usage

First install [CH340 driver](https://www.wch.cn/downloads/CH341SER_EXE.html) for your platform.

I found the official driver on Linux to not work at all, I used [this](https://github.com/juliagoda/CH341SER) intead.

Then add your user to the `dialout` group (if on Linux) to allow access to the serial port without root privileges. This is typically done with the following command:

```bash
sudo usermod -a -G dialout $USER
```

Usually you need to log out and log back in for the group change to take effect. Sometimes it may also be necessary to restart your computer.

Then find the serial port of your OYMotion hand. On Linux, you can use `ls /dev/ttyUSB*` to list all serial devices. The OYMotion hand is usually listed as `/dev/ttyUSB0` or similar.

Then install this package:

```bash
cd oymotion_rohand
pip install -e .
```

Then you can use the `RohandActor` class to control the OYMotion hand:

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld
from unienv_rohand import RohandActor
import numpy as np
import time

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04, # Usualy we should set this to exactly the same as the control timestep
    batch_size=None # None means single instance
)

actor = RohandActor(
    world,
    port="/dev/ttyUSB0",
    node_id=2,
    baudrate=115200,
    control_mode='position', # Use "position" or "angle" control mode
    control_timestep=0.04,
    update_timestep=0.04,
)

rng = np.random.default_rng(42)
actor.reset()

time_s = time.monotonic()

while True:
    dt = time.monotonic() - time_s
    actor.update(dt)
    obs = actor.get_observation()
    print(obs)
    rng, action = actor.action_space.sample(rng)
    actor.set_next_action(action)
    actor.pre_environment_step(dt)
    time_s += dt
```