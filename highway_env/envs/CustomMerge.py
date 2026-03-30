

import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import (
    AbstractLane,
    CircularLane,
    LineType,
    SineLane,
    StraightLane,
)
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import LaneIndex, RoadNetwork, Route
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle




class CustomMergeEnv(AbstractEnv):

    @classmethod
    def default_config(cls, num_agents = 3) -> dict:

        config = super().default_config()  # parentesi!
        config.update({
            "observation": { 
                "type": "MultiAgentObservation",
                "observation_config": { 
                    "type": "Kinematics"
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction"
                }
            },
            "controlled_vehicles": num_agents,

            #rewards
            "collision_reward": -5,
            "high_speed_reward": 0.4,
            "reward_speed_range": [7.0, 9.0],
            "arrived_reward": 1,
            "right_lane_reward": 0.1,  
            "lane_change_reward": 0,
            "merging_speed_reward": -0.5,

            #others
            
            "normalize_reward": False,
            "offroad_terminal": False,
            "destination": "o1",
            "other_vehicles_destinations": [
                "o1", "o2", "sxs", "sxr", "exs", "exr", "nxs", "nxr",
            ],

            "initial_vehicle_count": 4,
            "spawn_probability": 0.1,

            "duration": 60,

            #graphics
            "screen_width": 1200,
            "screen_height": 1200,
        })
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
                [0, 1],
            )
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> dict[str, float]:
        """Per-agent per-objective reward signal."""
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": vehicle.crashed,
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "arrived_reward": self.has_arrived(vehicle),
            "on_road_reward": vehicle.on_road,
        }




    def _is_terminated(self):
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
            or (self.config["offroad_terminal"] and not self.vehicle.on_road)
        )
    
    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        return vehicle.crashed or self.has_arrived(vehicle)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]





    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info
    


    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return (
            "c" in vehicle.lane_index[0]
            and "d" in vehicle.lane_index[1]
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
        )
    
    
    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        )
        info["agents_terminated"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )

        info["crashed"] = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        
        
        info["all_arrived"] = all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles)
        return info

    

        



    def _make_road(self) -> None:
        net = RoadNetwork()

        original_get_lane = net.get_lane

        
        def safe_get_lane(index):
            _from, _to, _id = index
            try:
                return original_get_lane(index)
            except IndexError:
                #return closes lane to the left if IndexError occurs
                return original_get_lane((_from, _to, -1))

        net.get_lane = safe_get_lane

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

      
        x, y = self._add_straight_highway_with_merging(road, 0, 0)


        self.road = road


    def _add_intersection(self, road: RegulatedRoad, x_start, y_start) -> tuple[int, int]:
        net = road.network
        
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = rotation @ np.array(
                [lane_width / 2, access_length + outer_distance]
            )
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10.0
                ),
            )
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10.0,
                ),
            )
            # Left turn
            l_center = rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10.0,
                ),
            )
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10.0
                ),
            )
            # Exit
            start = rotation @ np.flip(
                [lane_width / 2, access_length + outer_distance], axis=0
            )
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10.0
                ),
            )

        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

        return 0, 0
    
    @staticmethod
    def _add_straight_highway_with_merging(road: RegulatedRoad, x_start, y_start) -> tuple[int, int]:
        """
        Generates a straight highway with a merging lane.

        Note: The geometry and lane configuration logic is heavily inspired by 
        and adapted from the open-source repository `highway-agent-401` by ece1508-ai-alchemist:
        https://github.com/ece1508-ai-alchemist/highway-agent-401/blob/main/src/environments/highway401.py#L154
        """
        net = road.network
        lane_start = [x_start, y_start]
        merging_line_vertical_distance = 15
        lane_width = AbstractLane.DEFAULT_WIDTH
        line_length = 150
        amplitude = 4

        merging_straight_line_length = line_length - 50
        merging_sine_line_length = 39

        merging_straight_line2_length = merging_straight_line_length

        
        ab_0 = StraightLane(
            [lane_start[0], lane_start[1]],
            [lane_start[0] + line_length, lane_start[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        ab_1 = StraightLane(
            [lane_start[0], lane_start[1] + lane_width],
            [lane_start[0] + line_length, lane_start[1] + lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
        )

        net.add_lane("a", "b", ab_0)
        net.add_lane("a","b",ab_1,)

        # m1 straight merging line
        m12 = StraightLane(
            [lane_start[0], lane_start[1] + merging_line_vertical_distance],
            [
                lane_start[0] + merging_straight_line_length,
                lane_start[1] + merging_line_vertical_distance,
            ],
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
            forbidden=True,
        )
        # m2 sin merging line to b
        m23_end = merging_straight_line_length + merging_sine_line_length
        m23 = SineLane(
            [
                lane_start[0] + merging_straight_line_length,
                merging_line_vertical_distance - amplitude + lane_start[1],
            ],
            [
                lane_start[0] + m23_end,
                merging_line_vertical_distance - amplitude + lane_start[1],
            ],
            amplitude,
            2 * np.pi / (2 * -50),
            np.pi / 2,
            line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS],
            forbidden=True,
            priority=3,
        )

        net.add_lane("m1","m2",m12,)
        net.add_lane("m2", "b", m23)

        # bc straight merging line 2
        merging_line_end = m23_end + merging_straight_line2_length

        # bc straight line 0
        bc_0 = StraightLane(
            [lane_start[0] + line_length, lane_start[1]],
            [lane_start[0] + merging_line_end, lane_start[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        bc_1 = StraightLane(
            [lane_start[0], lane_start[1] + lane_width],
            [lane_start[0] + merging_line_end, lane_start[1] + lane_width],
            line_types=(LineType.NONE, LineType.STRIPED),
        )
        m34 = StraightLane(
            [lane_start[0] + m23_end, 2 * lane_width + lane_start[1]],
            [lane_start[0] + merging_line_end, 2 * lane_width + lane_start[1]],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
            forbidden=True,
            priority=1,
        )

        net.add_lane("b", "c", bc_0)
        net.add_lane("b","c",bc_1,) 
        net.add_lane("b", "c", m34)

        # cd straight line 0
        cd_0 = StraightLane(
            [lane_start[0] + merging_line_end, lane_start[1]],
            [lane_start[0] + merging_line_end + line_length, lane_start[1]],
            line_types=(LineType.CONTINUOUS, LineType.STRIPED),
        )
        cd_1 = StraightLane(
            [lane_start[0] + merging_line_end, lane_start[1] + lane_width],
            [lane_start[0] + merging_line_end + line_length, lane_start[1] + lane_width],
            line_types=(LineType.NONE, LineType.CONTINUOUS),
        )

        
        net.add_lane("c", "d", cd_0)
        net.add_lane("c", "d", cd_1)

        road.network = net
        road.objects.append(
            Obstacle(
                road,
                [
                    lane_start[0] + m23_end + merging_straight_line2_length,
                    lane_start[1] + 2 * lane_width,
                ],
            )
        )
        return lane_start[0] + merging_line_end + line_length, int(
            lane_start[1] + lane_width
        )
    

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        determines starting position and destination of ego-vehicles
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
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
        self._spawn_vehicle(
            60,
            spawn_probability=1.0,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0.0,
        )

        # Controlled vehicles
        self.controlled_vehicles = []
        
        
        # ("a", "b", 0) -> Autostrada principale, corsia di destra
        # ("a", "b", 1) -> Autostrada principale, corsia di sinistra
        # ("m1", "m2", 0) -> Corsia di immissione (merging)
        starting_lanes = [("a", "b", 0), ("a", "b", 1), ("m1", "m2", 0)]

        for ego_id in range(0, self.config["controlled_vehicles"]):
            
            
            random_lane_index = self.np_random.choice(len(starting_lanes))
            random_lane_tuple = starting_lanes[random_lane_index]
            
            ego_lane = self.road.network.get_lane(random_lane_tuple)
            
            
            
            #spawn ego wehicles at a distance of 10 m to avoid spawning over eachother
            longitudinal_position = 10.0 + (ego_id * 20.0)
            
            random_longitudinal_position = self.np_random.uniform(-2.0, 2.0) + longitudinal_position

            
            destination = "d" #node at end of the road

            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(random_longitudinal_position, 0.0), 
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(random_longitudinal_position),
            )
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            
            #prevent immediate collision at spawn
            for v in list(self.road.vehicles):  
                if (
                    v is not ego_vehicle
                    and np.linalg.norm(v.position - ego_vehicle.position) < 15 
                ):
                    if v not in self.controlled_vehicles:
                        self.road.vehicles.remove(v)


    

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight=True,
    ) -> None:
        """
        determines spawn position for non-ego vehicles
        """
        if self.np_random.uniform() > spawn_probability:
            return

        
        # ("a", "b", 0) -> Autostrada principale destra
        # ("a", "b", 1) -> Autostrada principale sinistra
        # ("m1", "m2", 0) -> Rampa di immissione
        spawn_lanes = [("a", "b", 0), ("a", "b", 1), ("m1", "m2", 0)]
        
        

        random_lane_index = self.np_random.choice(len(spawn_lanes))
        random_lane_tuple = spawn_lanes[random_lane_index]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            random_lane_tuple, # Usiamo la nuova corsia scelta
            longitudinal=(
                longitudinal + 5.0 + self.np_random.normal() * position_deviation
            ),
            speed=8.0 + self.np_random.normal() * speed_deviation,
        )
        
        
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
                
        vehicle.plan_route_to("d")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        
        return vehicle
    
    def _clear_vehicles(self) -> None:
        # Un veicolo sta "uscendo" dalla mappa se si trova nel tratto finale ("c" -> "d")
        # ed è arrivato quasi alla fine della corsia
        is_leaving = (
            lambda vehicle: vehicle.lane_index[0] == "c"
            and vehicle.lane_index[1] == "d"
            and vehicle.lane.local_coordinates(vehicle.position)[0]
            >= vehicle.lane.length - 4 * vehicle.LENGTH
        )
        
        self.road.vehicles = [
            vehicle
            for vehicle in self.road.vehicles
            if vehicle in self.controlled_vehicles
            or not (is_leaving(vehicle) or vehicle.route is None)
        ]




