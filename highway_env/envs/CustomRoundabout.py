from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, SineLane, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle


class CustomRoundaboutEnv(AbstractEnv):
    """
        this enviroment enhances intersection-v1 by allowing users to decide the spawn point and destination of each controlled vehicle
        "spawn_points: int[]" the values allowed are 0, 1, 2, 3, meaning respectivley: spawn- South(0), West(1), North(2), East(3)
        "multi_destinations: str[]" set the destination of each agent, values allowed are the outer nodes ("o0", "o1", "o2", "o3") 

        ISSUES:
            - when 2 agents have the same spawn point, there is currently no script offsetting their spawn position, so they crash at step 0
            - Challenger Vehicle is deleted if a agent spawns in its lane

    """
    ACTIONS: dict[int, str] = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": False,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "duration": 13,  # [s]

                "spawn_points": None,

                "destination": "o1",
                "multi_destinations": None,

                "controlled_vehicles": 2,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,

                "ego_vehicle_speed_limit": 9,
                "initial_simulation_steps": 3,
                "stopped_penalty": -0.1
                
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> dict[str, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [
            self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards)
            / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward = self.config["arrived_reward"] if rewards["arrived_reward"] else reward
        reward *= rewards["on_road_reward"]
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["arrived_reward"]],
                [-1, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        stopped_signal = 0
        speeding_signal = 0
        tailgating_signal = 0

        if vehicle.speed < 0.01:
            stopped_signal = 1.0 

        #
        if vehicle.speed > self.config["ego_vehicle_speed_limit"]:
            speeding_signal = 1.0 


        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(vehicle, lane_index=vehicle.lane_index)
        
        if front_vehicle is not None:
        
                distance = np.linalg.norm(vehicle.position - front_vehicle.position)
                
                if distance < 15.0:
                    relative_speed = vehicle.speed - front_vehicle.speed
                    if relative_speed > 0: 
                 
                        danger_factor = (15.0 - distance) / 15.0 
                        tailgating_signal = danger_factor

        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )

        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
            "stopped_penalty": stopped_signal,
            "speeding_penalty": speeding_signal,
            "tailgating_penalty": tailgating_signal,
            "step_penalty": 1
        }

    def _is_terminated(self) -> bool:
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)

        info["agents_rewards"] = tuple(
            self._agent_reward(action[i], vehicle) for i, vehicle in enumerate(self.controlled_vehicles)
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )

        info["speed"] = tuple(
            vehicle.speed for vehicle in self.controlled_vehicles
        )

        info["rewards"] = tuple(
            self._agent_rewards(action[i], vehicle) for i, vehicle in enumerate(self.controlled_vehicles)
        )

        info["crashed"] = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        
        
        info["all_arrived"] = all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) and not info["crashed"]

        info["episode_truncated"] = self._is_truncated()

        return info
    
    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        
        for vehicle in self.controlled_vehicles:
            if self.has_arrived(vehicle) and vehicle in self.road.vehicles:
                self.road.vehicles.remove(vehicle)

        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        net.add_lane(
            "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2],
                [2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [-2 - a, -dev / 2 + delta_en],
                [-2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        net.add_lane(
            "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en, 2 + a],
                [dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        net.add_lane(
            "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a, dev / 2 - delta_en],
                [2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        net.add_lane(
            "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a],
                [-dev / 2 + delta_st, 2 + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en, -2 - a],
                [-dev / 2, -2 - a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the Intersection Lanes

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        #EGO_VEHICLE_SPEED_LIMIT = self.config["ego_vehicle_speed_limit"] 
        DIRECTIONS = ["s", "e", "n", "w"]

        # Random vehicles
        simulation_steps = self.config["initial_simulation_steps"]  
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Challenger vehicle
        if self.config["disable_challenger_vehicle"] == False:
            self.challenger_vehicle = self._spawn_vehicle(
                60,
                spawn_probability=1.0,
                go_straight=True,
                position_deviation=0.1,
                speed_deviation=0.0,
            )

        # Controlled vehicles
        self.controlled_vehicles = []
        spawn_counts = {}
        
        for ego_id in range(0, self.config["controlled_vehicles"]):

            dest_options = [0, 1, 2, 3]

            if self.config["randomize_spawn_points"] == True:
                print("if")
                random_spawn_index= self.np_random.integers(0,4)
                dest_options.remove(random_spawn_index)
                d_in = DIRECTIONS[random_spawn_index]

                lane_index = (f"{d_in}er",
                              f"{d_in}es",
                              0) 

            elif self.config["spawn_points"] is not None:
                print("elif")
                spawn_points_list = self.config["spawn_points"]
                spawn_idx = spawn_points_list[ego_id % len(spawn_points_list)]
                d_in = DIRECTIONS[spawn_idx]
                
                lane_index = (f"{d_in}er",
                              f"{d_in}es",
                              0)
                
            else:
                print("else")
                spawn_idx = ego_id % 4
                d_in = DIRECTIONS[spawn_idx]
                lane_index = (f"{d_in}er", f"{d_in}es", 0)

            ego_lane = self.road.network.get_lane(lane_index)
            # Offset spawn position if multiple vehicles are in the same lane
            spawn_counts[lane_index] = spawn_counts.get(lane_index, 0) + 1
            longitudinal = 40.0 - (spawn_counts[lane_index] - 1) * 20


            if self.config["randomize_destinations"] == True:
                random_destination = self.np_random.choice(dest_options)
                d_out = DIRECTIONS[random_destination]
                destination = f"{d_out}xr" 

            elif self.config["multi_destinations"] is not None:
                dest_list = self.config["multi_destinations"]
                dest_idx = dest_list[ego_id % len(dest_list)] 
                d_out = DIRECTIONS[dest_idx]
                destination = f"{d_out}xr"

            else:
                d_out = DIRECTIONS[self.np_random.integers(0, 4)]
                destination = self.config["destination"] or f"{d_out}xr"

            
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(longitudinal + 5.0 * self.np_random.normal(1.0), 0.0),
                speed= ego_lane.speed_limit, #EGO_VEHICLE_SPEED_LIMIT,   #
                heading=ego_lane.heading_at(longitudinal),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(
                    ego_lane.speed_limit #EGO_VEHICLE_SPEED_LIMIT# 
                )
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if (
                    v is not ego_vehicle
                    and v not in self.controlled_vehicles
                    and v is not self.challenger_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 20
                ):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        DIRECTIONS = ["s", "e", "n", "w"]
        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        d_in = DIRECTIONS[route[0]]
        d_out = DIRECTIONS[route[1]]

        vehicle = vehicle_type.make_on_lane(
            self.road,
            (f"{d_in}er", f"{d_in}es", 0),
            longitudinal=(
                longitudinal + 5.0 + self.np_random.normal() * position_deviation
            ),
            speed=8.0 + self.np_random.normal() * speed_deviation,
        )
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to(f"{d_out}xr")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = (
            lambda vehicle: vehicle.lane_index[1] in ["sxr", "exr", "nxr", "wxr"]
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            vehicle.lane_index[1] in ["sxr", "exr", "nxr", "wxr"]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
    

#RENDERING 


    def render(self):
        surface = super().render()
        return surface




class MultiAgentRoundaboutEnv(CustomRoundaboutEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "lateral": False,
                        "longitudinal": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "controlled_vehicles": 2,
            }
        )
        return config

