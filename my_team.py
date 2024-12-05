from capture_agents import CaptureAgent
import distance_calculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearest_point

# from contest.capture_agents import CaptureAgent
# import contest.distance_calculator import distance_calculator
# import random
# import time
# import sys
# from contest.game import Directions
# import contest.game
# from contest.util import nearest_point
# import random
# import contest.util as util




#################
# Team Creation #
#################
def create_team(primary_agent_idx, secondary_agent_idx, is_red_team,
                primary_agent='OffensiveMCTAgent', secondary_agent='DefensiveMCTAgent'):
    """
    Initializes the team with two agents.

    Args:
        primary_agent_idx (int): Index for the primary agent.
        secondary_agent_idx (int): Index for the secondary agent.
        is_red_team (bool): Indicates if the team is red.
        primary_agent (str): Class name of the primary agent.
        secondary_agent (str): Class name of the secondary agent.

    Returns:
        list: A list containing instances of the two agents.
    """
    return [eval(primary_agent)(primary_agent_idx), eval(secondary_agent)(secondary_agent_idx)]


##########
# Agents #
##########

class MCTCaptureAgent(CaptureAgent):
    """
    Base class for agents using Monte Carlo Tree Search (MCT).
    """

    def register_initial_state(self, game_state):
        """
        Registers the initial game state.

        Args:
            game_state: The initial game state.
        """
        CaptureAgent.register_initial_state(self, game_state)
        # Unused variable for potential future use
        unused_variable = None

    def choose_action(self, game_state):
        """
        Selects an action randomly from the list of legal actions.

        Args:
            game_state: The current game state.

        Returns:
            Action: The chosen action.
        """
        possible_actions = game_state.get_legal_actions(self.index)
        return random.choice(possible_actions)

    def get_successor(self, game_state, action):
        """
        Generates the successor game state after performing the given action.

        Args:
            game_state: The current game state.
            action: The action to perform.

        Returns:
            game_state: The successor game state.
        """
        next_state = game_state.generate_successor(self.index, action)
        current_position = next_state.get_agent_state(self.index).get_position()

        if current_position != nearest_point(current_position):
            return next_state.generate_successor(self.index, action)
        else:
            return next_state

    def evaluate(self, game_state, action):
        """
        Evaluates the given action based on features and weights.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            float: The evaluation score.
        """
        feature_values = self.get_features(game_state, action)
        feature_weights = self.get_weights(game_state, action)
        return feature_values * feature_weights

    def get_features(self, game_state, action):
        """
        Extracts features for the evaluation function.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            util.Counter: A counter of feature values.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)

        # Placeholder for additional features
        temp_value = 0  # This variable is not used elsewhere

        return features

    def get_weights(self, game_state, action):
        """
        Defines weights for each feature.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            dict: A dictionary of feature weights.
        """
        return {'successor_score': 1.0}


class OffensiveMCTAgent(MCTCaptureAgent):
    """
    Offensive agent that uses Monte Carlo Tree Search to collect food and return home.
    """

    def register_initial_state(self, game_state):
        """
        Initializes the agent's starting position and calculates home points.

        Args:
            game_state: The initial game state.
        """
        super().register_initial_state(game_state)
        self.start_pos = game_state.get_initial_agent_position(self.index)

        # Determine home side points based on team color
        self.home_points = []
        if self.red:
            central_x = (game_state.data.layout.width - 2) // 2
        else:
            central_x = ((game_state.data.layout.width - 2) // 2) + 1

        for y in range(1, game_state.data.layout.height - 1):
            if not game_state.has_wall(central_x, y):
                self.home_points.append((central_x, y))

        self.food_timer = 0
        self.move_counter = 0
        self.start_location = game_state.get_agent_position(self.index)

        # Initialize survival and power mode attributes
        self.safe_point = self.start_location
        self.entry_points = self.home_points
        self.is_surviving = False
        self.is_powering = False
        self.total_food = len(self.get_food(game_state).as_list())

        self.opponent_red = not self.red
        if self.opponent_red:
            ghost_x, ghost_y = 1, 1
        else:
            ghost_x = game_state.data.layout.width - 2
            ghost_y = game_state.data.layout.height - 2

        self.ghost_position = (ghost_x, ghost_y)

        # Additional attribute for future use
        self.extra_attribute = []

    def choose_action(self, game_state):
        """
        Selects the best action using Monte Carlo Tree Search.

        Args:
            game_state: The current game state.

        Returns:
            Action: The chosen action.
        """
        remaining_food = len(self.get_food(game_state).as_list())
        carried_food = game_state.get_agent_state(self.index).num_carrying

        if self.get_previous_observation() is not None:
            previous_carried = self.get_previous_observation().get_agent_state(self.index).num_carrying
        else:
            previous_carried = 0

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [enemy for enemy in enemies if
                         not enemy.is_pacman and enemy.get_position() and enemy.scared_timer == 0]

        if previous_carried == carried_food and not active_ghosts:
            self.food_timer += 1
        else:
            self.food_timer = 0

        # Monte Carlo Tree Search parameters
        ITERATIONS = 32
        SIMULATION_DEPTH = 10

        start_time = time.time()

        # Update survival and power modes
        self.update_survival_mode(game_state)
        self.update_power_mode(game_state)

        if not game_state.get_agent_state(self.index).is_pacman:
            self.safe_point = self.start_location

        if not self.is_surviving:
            mct_root = self.construct_mct(game_state, ITERATIONS, SIMULATION_DEPTH)
            child_nodes = mct_root.children
            evaluation_scores = [child.reward / child.count for child in child_nodes]
            highest_score = max(evaluation_scores)
            best_child = [c for c, v in zip(child_nodes, evaluation_scores) if v == highest_score][0]

            print(f'Evaluation time for agent {self.index}: {time.time() - start_time:.4f}')
            self.move_counter += 1
            return best_child.game_state.get_agent_state(self.index).configuration.direction
        else:
            chosen_action = self.navigate_home(game_state)
            print(f'Evaluation time for agent {self.index}: {time.time() - start_time:.4f}')
            self.move_counter += 1
            if chosen_action is None:
                available_actions = game_state.get_legal_actions(self.index)
                return random.choice(available_actions)
            else:
                return chosen_action

    def update_survival_mode(self, game_state):
        """
        Determines whether the agent should enter survival mode based on current conditions.

        Args:
            game_state: The current game state.
        """
        self.is_surviving = False
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_ghosts = [enemy for enemy in enemies if
                            not enemy.is_pacman and enemy.get_position() and enemy.scared_timer < 5]
        agent_state = game_state.get_agent_state(self.index)
        agent_pos = agent_state.get_position()

        from distance_calculator import manhattan_distance
        ghost_maze_distance = 9999
        ghost_manhattan_distance = 9999
        if dangerous_ghosts:
            ghost_maze_distance = min(
                [self.get_maze_distance(agent_pos, ghost.get_position()) for ghost in dangerous_ghosts])
            ghost_manhattan_distance = min(
                [manhattan_distance(agent_pos, ghost.get_position()) for ghost in dangerous_ghosts])

        food_carried = agent_state.num_carrying
        remaining_food = len(self.get_food(game_state).as_list())

        if agent_state.is_pacman and food_carried > (self.total_food / 2 - 1) and dangerous_ghosts:
            if ghost_manhattan_distance <= 5:
                self.is_surviving = True
                self.safe_point = self.start_location
        elif agent_state.is_pacman and food_carried > 2 and dangerous_ghosts:
            if food_carried > (self.total_food / 4):
                self.is_surviving = True
            elif ghost_maze_distance <= 5:
                self.is_surviving = True
        elif remaining_food <= 2:
            self.is_surviving = True
            self.safe_point = self.start_location
        elif food_carried > 0 and self.move_counter > 270:
            self.is_surviving = True
            self.safe_point = self.start_location

        if self.is_surviving and dangerous_ghosts:
            ghost_positions = [enemy.get_position() for enemy in dangerous_ghosts]
            distance_to_home = self.get_maze_distance(agent_pos, self.safe_point)
            ghost_to_home_distance = min([self.get_maze_distance(gp, self.safe_point) for gp in ghost_positions])

            # Validate the safe point
            if ghost_to_home_distance < distance_to_home:
                for home_pt in self.home_points:
                    distance_to_home = self.get_maze_distance(agent_pos, home_pt)
                    ghost_to_home_distance = min([self.get_maze_distance(gp, home_pt) for gp in ghost_positions])
                    if distance_to_home < ghost_to_home_distance:
                        self.safe_point = home_pt
                        break

    def update_power_mode(self, game_state):
        """
        Updates the power mode status based on the opponents' scared timers.

        Args:
            game_state: The current game state.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        min_scared_timer = min([enemy.scared_timer for enemy in enemies])
        if min_scared_timer < 15:
            self.is_powering = False
        else:
            self.is_powering = True
            self.is_surviving = False
            self.safe_point = self.start_location

    def go_to_capsule(self, game_state):
        """
        Determines the action to move towards a capsule while avoiding ghosts.

        Args:
            game_state: The current game state.

        Returns:
            Action: The chosen action towards a capsule.
        """
        available_actions = game_state.get_legal_actions(self.index)
        optimal_distance = 9999
        selected_action = None
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_positions = [enemy.get_position() for enemy in enemies if
                           not enemy.is_pacman and enemy.get_position() and enemy.scared_timer == 0]

        # Eliminate actions leading to dead ends
        for act in available_actions[:]:
            if self.is_dead_end(game_state, act, 20):
                available_actions.remove(act)

        for act in available_actions:
            successor = self.get_successor(game_state, act)
            if self.get_capsules(successor) != self.get_capsules(game_state):
                return act
            agent_pos = successor.get_agent_position(self.index)
            capsules = self.get_capsules(successor)
            min_dist = 9999
            closest_cap = (1, 1)
            for cap in capsules:
                distance = self.get_maze_distance(agent_pos, cap)
                if distance < min_dist:
                    min_dist = distance
                    closest_cap = cap

            distance_to_cap = self.get_maze_distance(agent_pos, closest_cap)
            if ghost_positions:
                if agent_pos in ghost_positions or agent_pos == self.start_location:
                    distance_to_cap += 99999999
                elif min([self.get_maze_distance(agent_pos, gp) for gp in ghost_positions]) < 2:
                    distance_to_cap += 99999999
            if distance_to_cap < optimal_distance:
                selected_action = act
                optimal_distance = distance_to_cap
        return selected_action

    def navigate_home(self, game_state):
        """
        Determines the action to safely return home.

        Args:
            game_state: The current game state.

        Returns:
            Action: The chosen action to return home.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        shortest_distance = float('inf')
        chosen_action = None
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_positions = [enemy.get_position() for enemy in enemies if
                           not enemy.is_pacman and enemy.get_position() and enemy.scared_timer < 5]

        # Prioritize capsules over ghosts
        capsules = self.get_capsules(game_state)
        if capsules and ghost_positions:
            for cap in capsules:
                ghost_to_cap_distance = min([self.get_maze_distance(cap, gp) for gp in ghost_positions])
                agent_to_cap_distance = self.get_maze_distance(game_state.get_agent_position(self.index), cap)
                if agent_to_cap_distance < ghost_to_cap_distance:
                    return self.go_to_capsule(game_state)

        # Remove actions leading to dead ends
        for act in legal_actions[:]:
            if self.is_dead_end(game_state, act, 20):
                legal_actions.remove(act)

        for act in legal_actions:
            successor = self.get_successor(game_state, act)
            agent_pos = successor.get_agent_position(self.index)
            distance = self.get_maze_distance(self.safe_point, agent_pos)

            if ghost_positions:
                if agent_pos in ghost_positions or agent_pos == self.start_location:
                    distance += 99999999
                elif min([self.get_maze_distance(agent_pos, gp) for gp in ghost_positions]) < 2:
                    distance += 99999999
            if distance < shortest_distance:
                chosen_action = act
                shortest_distance = distance
        return chosen_action

    def is_dead_end(self, game_state, action, depth):
        """
        Recursively checks for dead ends based on a depth limit.

        Args:
            game_state: The current game state.
            action: The action to evaluate.
            depth (int): Depth limit for recursion.

        Returns:
            bool: True if it's a dead end, False otherwise.
        """
        if depth == 0:
            return False
        successor = game_state.generate_successor(self.index, action)
        legal_actions = successor.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        current_direction = successor.get_agent_state(self.index).configuration.direction
        reverse_direction = Directions.REVERSE[current_direction]
        if reverse_direction in legal_actions:
            legal_actions.remove(reverse_direction)

        if not legal_actions:
            return True
        for next_act in legal_actions:
            if not self.is_dead_end(successor, next_act, depth - 1):
                return False
        return True

    def is_empty_path(self, game_state, action, depth):
        """
        Checks for paths without any food based on a depth limit.

        Args:
            game_state: The current game state.
            action: The action to evaluate.
            depth (int): Depth limit for recursion.

        Returns:
            bool: True if the path is empty, False otherwise.
        """
        if depth == 0:
            return False
        successor = game_state.generate_successor(self.index, action)
        current_score = game_state.get_agent_state(self.index).num_carrying
        new_score = successor.get_agent_state(self.index).num_carrying

        capsules = self.get_capsules(game_state)
        agent_pos = successor.get_agent_position(self.index)

        if agent_pos in capsules:
            return False

        if current_score < new_score:
            return False

        legal_actions = successor.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        current_direction = successor.get_agent_state(self.index).configuration.direction
        reverse_direction = Directions.REVERSE[current_direction]

        if reverse_direction in legal_actions:
            legal_actions.remove(reverse_direction)

        if not legal_actions:
            return True
        for next_act in legal_actions:
            if not self.is_empty_path(successor, next_act, depth - 1):
                return False
        return True

    def construct_mct(self, game_state, iter_times, simulate_depth):
        """
        Constructs the Monte Carlo Tree based on the current game state.

        Args:
            game_state: The current game state.
            iter_times (int): Number of iterations for MCT.
            simulate_depth (int): Depth for simulations.

        Returns:
            Node: The root node of the constructed MCT.
        """
        start_time = time.time()
        root = Node(game_state, 0.0, 0)
        successors = self.generate_successors(root.game_state)

        for suc in successors:
            child = Node(suc, 0.0, 0)
            root.add_child(child)

        for _ in range(iter_times):
            current_node = root
            if time.time() - start_time > 0.95:
                break
            # Selection
            while current_node.children:
                uct_values = [self.calculate_uct(child, child.parent) for child in current_node.children]
                max_uct = max(uct_values)
                best_children = [c for c, v in zip(current_node.children, uct_values) if v == max_uct]
                if best_children:
                    current_node = random.choice(best_children)
                    continue
                break
            # Expansion
            if current_node.count != 0:
                child_successors = self.generate_successors(current_node.game_state)
                for suc in child_successors:
                    new_child = Node(suc, 0.0, 0)
                    current_node.add_child(new_child)
                if current_node.children:
                    current_node = random.choice(current_node.children)
                else:
                    continue
            # Simulation
            simulation_reward = self.run_simulation(current_node.game_state, simulate_depth, 0.0)
            # Backpropagation
            while current_node.parent is not None:
                current_node.reward += simulation_reward
                current_node.count += 1
                current_node = current_node.parent
            # Update root
            root.reward += simulation_reward
            root.count += 1

        return root

    def generate_successors(self, game_state):
        """
        Generates all valid successor states excluding STOP actions and empty paths.

        Args:
            game_state: The current game state.

        Returns:
            list: A list of successor game states.
        """
        possible_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in possible_actions:
            possible_actions.remove(Directions.STOP)
        for act in possible_actions[:]:
            if self.is_empty_path(game_state, act, 8):
                possible_actions.remove(act)
        return [self.get_successor(game_state, act) for act in possible_actions]

    def run_simulation(self, game_state, depth, cumulative_reward):
        """
        Simulates a random play from the given game state.

        Args:
            game_state: The current game state.
            depth (int): Depth limit for the simulation.
            cumulative_reward (float): Accumulated reward.

        Returns:
            float: The reward obtained from the simulation.
        """
        legal_moves = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_moves:
            legal_moves.remove(Directions.STOP)
        if not legal_moves:
            return cumulative_reward
        chosen_move = random.choice(legal_moves)

        successor_state = self.get_successor(game_state, chosen_move)
        remaining_depth = depth - 1
        cumulative_reward += self.evaluate(game_state, chosen_move)

        # Update ghost position based on current state
        if not successor_state.get_agent_state(self.index).is_pacman:
            ghost_distance = float('inf')
            opponents = [successor_state.get_agent_state(i) for i in self.get_opponents(successor_state)]
            for opp in opponents:
                if opp.get_position() is None:
                    continue
                if not opp.is_pacman:
                    distance = self.get_maze_distance(successor_state.get_agent_position(self.index),
                                                      opp.get_position())
                    if distance < ghost_distance:
                        ghost_distance = distance
                        self.ghost_position = opp.get_position()

        if remaining_depth > 0:
            return self.run_simulation(successor_state, remaining_depth, cumulative_reward)
        return cumulative_reward

    def calculate_uct(self, node, parent_node):
        """
        Calculates the UCT (Upper Confidence Bound) value for a node.

        Args:
            node (Node): The child node.
            parent_node (Node): The parent node.

        Returns:
            float: The UCT value.
        """
        import math
        exploration_factor = 0.75
        if node.count > 0:
            ucb = 2 * exploration_factor * math.sqrt(2 * math.log(parent_node.count) / node.count)
        else:
            ucb = float('inf')
        return node.reward + ucb

    def get_features(self, game_state, action):
        """
        Extracts features specific to the offensive agent.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            util.Counter: A counter of feature values.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        agent_state = successor.get_agent_state(self.index)
        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)
        features['successor_score'] = -len(food_list)
        features['eaten_capsules'] = -len(capsules)

        reverse_dir = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_dir:
            features['reverse'] = 1

        # Distance to the nearest food
        agent_pos = agent_state.get_position()
        if food_list:
            min_food_distance = min([self.get_maze_distance(agent_pos, food) for food in food_list])
            features['distance_to_food'] = min_food_distance

        # Distance to opponents
        features['distance_to_opponent'] = 0
        opponent_indices = self.get_opponents(game_state)
        for opp in opponent_indices:
            if not successor.get_agent_state(opp).is_pacman:
                opp_pos = successor.get_agent_position(opp)
                if opp_pos:
                    if self.get_maze_distance(agent_pos, opp_pos) == 1:
                        features['distance_to_opponent'] = -100
                    else:
                        features['distance_to_opponent'] = self.get_maze_distance(agent_pos, opp_pos)

        # Distance to the best entry point
        if not game_state.get_agent_state(self.index).is_pacman:
            best_entry_point = (0, 0)
            ghost_to_entry_distance = 0.0
            for entry in self.entry_points:
                distance = self.get_maze_distance(self.ghost_position, entry)
                if distance > ghost_to_entry_distance:
                    ghost_to_entry_distance = distance
                    best_entry_point = entry
            features['distance_to_entry'] = self.get_maze_distance(agent_pos, best_entry_point)

        # Distance to the closest home point in survival mode
        if self.is_surviving:
            min_home_distance = min([self.get_maze_distance(agent_pos, home_pt) for home_pt in self.home_points])
            features['distance_to_home'] = min_home_distance

        # Score when arriving home
        features['arrived_home'] = self.get_score(successor)

        # Check if the agent is at the start position
        features['is_dead'] = 0
        if agent_pos == self.start_pos:
            features['is_dead'] = 1

        # Check for dead ends
        features['dead_end'] = 0
        possible_actions = successor.get_legal_actions(self.index)
        if self.is_surviving and len(possible_actions) <= 1:
            features['dead_end'] = 1

        # Additional logic can be added here
        extra_logic = 42  # Unused variable for future enhancements

        return features

    def get_weights(self, game_state, action):
        """
        Defines weights for offensive agent features.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            dict: A dictionary of feature weights.
        """
        if self.is_powering:
            return {'successor_score': 150, 'distance_to_food': -10}
        return {
            'successor_score': 150,
            'distance_to_food': -5,
            'reverse': -3,
            'distance_to_entry': -10,
            'is_dead': -200,
            'dead_end': -100,
            'eaten_capsules': 200
        }


class DefensiveMCTAgent(MCTCaptureAgent):
    """
    Defensive agent that uses Monte Carlo Tree Search for strategic defense.
    """

    def register_initial_state(self, game_state):
        """
        Initializes the defensive agent's starting position and detection points.

        Args:
            game_state: The initial game state.
        """
        self.start_location = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)
        self.initial_defense_food = self.get_food_you_are_defending(game_state).as_list()
        self.chase_targets = []
        central_height = (game_state.data.layout.height - 2) // 2

        # Determine the central detection point based on team color
        if self.red:
            central_x = (game_state.data.layout.width - 2) // 2
            for x in range(central_x, 0, -1):
                if not game_state.has_wall(x, central_height):
                    self.detect_point = [(x, central_height)]
                    break
        else:
            central_x = ((game_state.data.layout.width - 2) // 2) + 1
            for x in range(central_x, game_state.data.layout.width):
                if not game_state.has_wall(x, central_height):
                    self.detect_point = [(x, central_height)]
                    break

        # Additional attribute for potential use
        self.additional_info = {}

    def choose_action(self, game_state):
        """
        Chooses the best defensive action based on evaluation scores.

        Args:
            game_state: The current game state.

        Returns:
            Action: The chosen action.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        action_values = [self.evaluate(game_state, act) for act in legal_actions]
        max_value = max(action_values)
        best_actions = [act for act, val in zip(legal_actions, action_values) if val == max_value]
        current_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        current_invaders = [enemy for enemy in current_enemies if enemy.is_pacman and enemy.get_position()]

        defending_food = self.get_food_you_are_defending(game_state).as_list()
        agent_pos = game_state.get_agent_state(self.index).get_position()

        if len(self.initial_defense_food) - len(defending_food) > 0:
            eaten_food = list(set(self.initial_defense_food).difference(set(defending_food)))
            self.initial_defense_food = defending_food
            self.chase_targets = eaten_food

        agent_state = game_state.get_agent_state(self.index)

        # Attack mode based on scared timer
        if agent_state.scared_timer > 10:
            attack_scores = [self.evaluate_attack(game_state, act) for act in legal_actions]
            max_attack_score = max(attack_scores)
            best_attack_actions = [act for act, val in zip(legal_actions, attack_scores) if val == max_attack_score]
            return random.choice(best_attack_actions)

        elif agent_state.scared_timer <= 10:
            if current_invaders:
                self.chase_targets = []
            elif self.chase_targets:
                path_to_food = self.find_path(game_state, self.chase_targets[0])
                if path_to_food:
                    return path_to_food[0]
                if agent_pos == self.chase_targets[0]:
                    self.chase_targets = []
            elif self.detect_point:
                search_path = self.find_path(game_state, self.detect_point[0])
                if search_path:
                    return search_path[0]
        return random.choice(best_actions)

    def find_path(self, game_state, destination):
        """
        Uses the A* algorithm to find a path to the destination.

        Args:
            game_state: The current game state.
            destination (tuple): The target position.

        Returns:
            list: A list of actions to reach the destination.
        """
        from util import PriorityQueue
        visited = set()
        action_paths = {}
        path_costs = {}
        start_pos = game_state.get_agent_state(self.index).get_position()
        action_paths[start_pos] = []
        path_costs[start_pos] = 0

        priority_queue = PriorityQueue()
        priority_queue.push(game_state, 0)

        while not priority_queue.is_empty():
            current_state = priority_queue.pop()
            current_loc = current_state.get_agent_state(self.index).get_position()

            if current_loc == destination:
                return action_paths[current_loc]

            if current_loc in visited:
                continue

            visited.add(current_loc)
            possible_actions = current_state.get_legal_actions(self.index)

            for act in possible_actions:
                successor_state = self.get_successor(current_state, act)
                successor_loc = successor_state.get_agent_state(self.index).get_position()
                new_cost = path_costs[current_loc] + 1

                if successor_loc not in path_costs or new_cost < path_costs[successor_loc]:
                    path_costs[successor_loc] = new_cost
                    action_paths[successor_loc] = action_paths[current_loc] + [act]
                    heuristic_val = self.heuristic_distance(successor_loc, destination)
                    priority = new_cost + heuristic_val
                    priority_queue.push(successor_state, priority)
        return []

    def heuristic_distance(self, loc, destination):
        """
        Calculates the Manhattan distance between two points.

        Args:
            loc (tuple): Current location.
            destination (tuple): Destination location.

        Returns:
            int: The Manhattan distance.
        """
        from util import manhattan_distance
        return manhattan_distance(loc, destination)

    def get_features(self, game_state, action):
        """
        Extracts features specific to the defensive agent.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            util.Counter: A counter of feature values.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        agent_state = successor.get_agent_state(self.index)
        agent_pos = agent_state.get_position()

        # Determine if the agent is on defense or offense
        features['on_defense'] = 1
        if agent_state.is_pacman:
            features['on_defense'] = 0
        if action == Directions.STOP:
            features['stop'] = 1

        # Distance to invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position()]
        features['num_invaders'] = len(invaders)
        if invaders:
            min_invader_distance = min([self.get_maze_distance(agent_pos, inv.get_position()) for inv in invaders])
            features['invader_distance'] = min_invader_distance
            if agent_state.scared_timer > 0 and min_invader_distance == 0:
                features['scared'] = 1
                features['stop'] = 0

        # Additional evaluation logic
        temp_feature = 10  # Placeholder variable

        return features

    def get_weights(self, game_state, action):
        """
        Defines weights for defensive agent features.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            dict: A dictionary of feature weights.
        """
        return {
            'num_invaders': -1000,
            'on_defense': 1000,
            'invader_distance': -10,
            'stop': -100,
            'scared': -1000
        }

    def evaluate_attack(self, game_state, action):
        """
        Evaluates actions in attack mode.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            float: The evaluation score.
        """
        attack_features = self.get_attack_features(game_state, action)
        attack_weights = self.get_attack_weights(game_state, action)
        return attack_features * attack_weights

    def get_attack_features(self, game_state, action):
        """
        Extracts features for attack mode.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            util.Counter: A counter of feature values.
        """
        features = util.Counter()
        if action == Directions.STOP:
            features['stop'] = 1

        successor = self.get_successor(game_state, action)
        succ_agent_state = successor.get_agent_state(self.index)
        succ_pos = succ_agent_state.get_position()
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position()]
        capsules = self.get_capsules(successor)

        # Distance to ghosts
        features['distance_to_ghost'] = float('inf')
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(succ_pos, ghost.get_position()) for ghost in ghosts])
            nearest_ghost = [ghost for ghost in ghosts if
                             self.get_maze_distance(succ_pos, ghost.get_position()) == min_ghost_dist]
            if nearest_ghost[0].scared_timer > 0:
                features['distance_to_ghost'] = float('inf')
            elif succ_agent_state.is_pacman:
                features['distance_to_ghost'] = min_ghost_dist

        # Capsules and food
        features['remaining_capsules'] = len(capsules)
        if capsules:
            features['distance_to_capsule'] = min([self.get_maze_distance(succ_pos, cap) for cap in capsules])

        food_remaining = self.get_food(successor).as_list()
        features['remaining_food'] = len(food_remaining)
        if food_remaining:
            features['distance_to_food'] = min([self.get_maze_distance(succ_pos, food) for food in food_remaining])

        # Additional logic
        extra_evaluation = "Not used"  # Placeholder string

        return features

    def get_attack_weights(self, game_state, action):
        """
        Defines weights for attack mode features.

        Args:
            game_state: The current game state.
            action: The action to evaluate.

        Returns:
            dict: A dictionary of feature weights.
        """
        return {
            'remaining_food': -100,
            'distance_to_food': -1,
            'remaining_capsules': -100,
            'distance_to_capsule': -1,
            'distance_to_ghost': 1000,
            'stop': -3000,
        }


class Node:
    """
    Represents a node in the Monte Carlo Tree.
    """

    def __init__(self, game_state, reward, visit_count, parent=None, children=None):
        """
        Initializes a new node.

        Args:
            game_state: The game state at this node.
            reward (float): Accumulated reward.
            visit_count (int): Number of times this node was visited.
            parent (Node, optional): Parent node. Defaults to None.
            children (list, optional): List of child nodes. Defaults to None.
        """
        self.game_state = game_state
        self.reward = reward
        self.count = visit_count
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child_node):
        """
        Adds a child node to the current node.

        Args:
            child_node (Node): The child node to add.
        """
        self.children.append(child_node)
        child_node.parent = self

    def display_node(self):
        """
        Prints the node's details.
        """
        print("Monte Carlo Node")
        print(self.game_state)
        print("Reward: ", self.reward, " Visit Count: ", self.count)

    def display_children(self):
        """
        Prints details of all child nodes.
        """
        for child in self.children:
            child.display_node()


# Additional class or function definitions can be placed here if needed
def additional_helper_function():
    """
    An additional helper function that is not used.
    """
    pass

