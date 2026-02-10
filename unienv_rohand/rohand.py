from typing import List, Literal, Optional, Dict

# Import register addresses and other constants
from . import roh_registers_v1 as roh_const
from . import roh_finger_status

from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import DictSpace, BoxSpace

class RohandActor(WorldNode[
    None, Dict[str, NumpyArrayType], NumpyArrayType,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    after_reset_priorities = {0}
    pre_environment_step_priorities = {0}
    post_environment_step_priorities = {0}

    joint_names = [
        "thumb_middle",
        "index_middle",
        "middle_middle",
        "ring_middle",
        "little_middle",
        "thumb_proximal",
    ]
    n_joints = len(joint_names)

    """
    Joint Angles in degrees:
    Completely Closed: 2.26, 100.22,  97.81, 101.38,  98.84, 89.98
    Completely Open: 36.7, 178.37, 176.06, 176.56, 174.86, 0
    """
    joint_limit_low_deg = np.array([0, 95, 95, 95, 95, 0], dtype=np.float32)  # Joint limits in degrees
    joint_limit_high_deg = np.array([40.0, 180.0, 180.0, 180.0, 180.0, 90.0], dtype=np.float32)  # Joint limits in degrees
    joint_limit_low = np.deg2rad(joint_limit_low_deg)  # Convert to radians
    joint_limit_high = np.deg2rad(joint_limit_high_deg)  # Convert to radians

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "rohand",
        port: str = "/dev/ttyUSB0",
        node_id: int = 2,
        baudrate: int = 115200,
        *,
        control_mode: Literal['angle', 'position'] = 'position',
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04, # The update frequency (background frequency to send commands and receive observations)
    ):
        # Initialize Communication
        self.port = port
        self.node_id = node_id
        self.baudrate = baudrate
        self.client = ModbusSerialClient(
            port,
            framer=FramerType.RTU,
            baudrate=baudrate,
        )
        if self.client.connect():
            print(f"Connected to ROH at {port} with NODE_ID {node_id}")
        else:
            raise ConnectionError(f"Failed to connect to ROH at {port}, use `ls /dev/ttyUSB*` if on linux to find the correct port.")

        # Set WorldNode-related attributes
        self.name = name
        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), "Control timestep must be a multiple of world timestep."
        self.world = world
        self.control_mode = control_mode
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {
                "joint_angles": BoxSpace( # Joint Angles in radians
                    NumpyComputeBackend,
                    low=self.joint_limit_low,
                    high=self.joint_limit_high,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_positions": BoxSpace( # Joint Positions in [-1, 1] range, -1 represents fully open, 1 represents fully closed
                    NumpyComputeBackend,
                    low=-1.0,
                    high=1.0,
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_currents": BoxSpace( # Joint Currents in Amperes
                    NumpyComputeBackend,
                    low=0.0,
                    high=1.178,  # Default value for `ROH_FINGER_CURRENT_LIMIT0` is 1178 mA
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
                "joint_forces": BoxSpace( # Joint Forces in Newtons
                    NumpyComputeBackend,
                    low=0.0,
                    high=15.0,  # Default value for `ROH_FINGER_FORCE_LIMIT0` is 15_000 mN
                    dtype=np.float32,
                    shape=(self.n_joints,),
                ),
            }
        )
        self.action_space = BoxSpace(
            NumpyComputeBackend,
            self.joint_limit_low if control_mode == 'angle' else -1.0,
            self.joint_limit_high if control_mode == 'angle' else 1.0,
            dtype=np.float32,
            shape=(self.n_joints,),
        )
        self.control_timestep = control_timestep  # Control timestep in seconds
        self.update_timestep = update_timestep
        self._current_observation : Optional[Dict[str, NumpyArrayType]] = None
        self._next_action : Optional[NumpyArrayType] = None

    # ========== Actor Implementation ==========
    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend
    
    @property
    def device(self) -> None:
        return None

    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        if self._next_action is not None:
            if self.control_mode == 'angle':
                self.send_angle_commands_safe(self._next_action)
            elif self.control_mode == 'position':
                self.send_position_commands_safe(self._next_action)
    
    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self._current_observation = {
            "joint_angles": self.read_finger_angles(),
            "joint_positions": self.read_finger_positions(),
            "joint_currents": self.read_finger_currents(),
            "joint_forces": self.read_finger_forces(),
        }

    def get_observation(self):
        return self._current_observation

    def set_next_action(self, action):
        assert isinstance(action, NumpyArrayType), "Action must be a numpy array."
        assert action.shape == (self.n_joints,), f"Action shape must be ({self.n_joints},), got {action.shape}"
        self._next_action = action
    
    def close(self):
        self.client.close()

    # ========== Helper Methods ==========

    def read_finger_angles(self) -> np.ndarray:
        # Convert angles to int16 and scale
        # In the demo they explicitly converted uint16 to int16 before sending
        # but from my inspection looks like angle values never exceed 65535 / 2?

        readings = self.read_registers(roh_const.ROH_FINGER_ANGLE0, count=self.n_joints)
        readings = np.asarray(readings).astype(np.int16) # This will explicitly convert uint16 values to int16
        return readings.astype(np.float32) / 100.0 * np.pi / 180.0  # Convert to radians

    def read_finger_positions(self) -> np.ndarray:
        readings = np.asarray(self.read_registers(roh_const.ROH_FINGER_POS0, count=self.n_joints), dtype=np.uint16)
        return readings.astype(np.float32) / 65535.0 * 2.0 - 1.0  # Convert to [-1, 1] range

    def read_finger_currents(self) -> np.ndarray:
        """
        Read the current of each finger in Ampheres.
        """
        readings = np.asarray(self.read_registers(roh_const.ROH_FINGER_CURRENT0, count=self.n_joints), dtype=np.uint16)
        return readings.astype(np.float32) / 1000.0  # Convert to Amperes

    def read_finger_forces(self) -> np.ndarray:
        """
        Read the force of each finger in Newtons.
        """
        readings = np.asarray(self.read_registers(roh_const.ROH_FINGER_FORCE0, count=self.n_joints), dtype=np.uint16)
        return readings.astype(np.float32) / 1000.0
    
    def read_finger_statuses(self) -> np.ndarray:
        """
        Read the status of each finger.
        Returns an array of roh_finger_status.FingerStatus.
        """
        readings = np.asarray(self.read_registers(roh_const.ROH_FINGER_STATUS0, count=self.n_joints), dtype=np.uint16)
        return readings

    def send_angle_commands(self, angles : np.ndarray) -> None:
        """
        Send raw angle commands to the RoHand without checking if fingers are stuck
        Angles should be in radians.
        """
        if angles.shape != (self.n_joints,):
            raise ValueError(f"Expected angles shape to be ({self.n_joints},), got {angles.shape}")
        
        # Convert angles to int16 and scale
        # In the demo they explicitly converted int16 to uint16 before sending
        # but from my inspection looks like angle values never exceed 65535 / 2?
        
        scaled_angles = np.round(angles / np.pi * 180.0 * 100.0).astype(np.int16)
        self.write_registers(roh_const.ROH_FINGER_ANGLE_TARGET0, scaled_angles.tolist())

    def send_angle_commands_safe(self, angles : np.ndarray) -> None:
        """
        Send angle commands to the RoHand and check if fingers are stuck.
        Angles should be in radians.
        """
        if angles.shape != (self.n_joints,):
            raise ValueError(f"Expected angles shape to be ({self.n_joints},), got {angles.shape}")
        
        current_finger_angles = self.read_joint_angles()
        current_finger_statuses = self.read_finger_statuses()
        target_finger_angles = np.where(
            current_finger_statuses == roh_finger_status.STATUS_STUCK,
            current_finger_angles, # If stuck, keep current angle
            angles # Otherwise, use target angle
        )
        self.send_angle_commands(target_finger_angles)

    def send_position_commands(self, positions : np.ndarray) -> None:
        """
        Send raw position commands to the RoHand without checking if fingers are stuck.
        Positions should be in [-1, 1] range.
        """
        if positions.shape != (self.n_joints,):
            raise ValueError(f"Expected positions shape to be ({self.n_joints},), got {positions.shape}")
        
        # Convert positions to uint16 and scale
        scaled_positions = ((positions + 1.0) / 2.0 * 65535.0).round().astype(np.uint16)
        self.write_registers(roh_const.ROH_FINGER_POS_TARGET0, scaled_positions.tolist())

    def send_position_commands_safe(self, positions : np.ndarray) -> None:
        """
        Send position commands to the RoHand and check if fingers are stuck.
        Positions should be in [-1, 1] range.
        """
        if positions.shape != (self.n_joints,):
            raise ValueError(f"Expected positions shape to be ({self.n_joints},), got {positions.shape}")
        
        current_finger_positions = self.read_joint_positions()
        current_finger_statuses = self.read_finger_statuses()
        target_finger_positions = np.where(
            current_finger_statuses == roh_finger_status.STATUS_STUCK,
            current_finger_positions, # If stuck, keep current position
            positions # Otherwise, use target position
        )
        self.send_position_commands(target_finger_positions)

    # ========== Register Read/Write Methods ==========
    
    def write_registers(self, address_start : int, values : List[int]) -> None:
        self.client.write_registers(
            address_start,
            values,
            slave=self.node_id
        )
    
    def write_single_register(self, address : int, value : int) -> None:
        """Write a single register to the ROH."""
        self.write_registers(address, [value])

    def read_registers(self, address_start : int, count : int = 1) -> List[int]:
        """Read a register from the ROH."""
        response = self.client.read_holding_registers(address_start, count=count, slave=self.node_id)
        return response.registers
    
    def read_single_register(self, address : int) -> int:
        """Read a single register from the ROH."""
        response = self.read_registers(address, count=1)
        return response[0]