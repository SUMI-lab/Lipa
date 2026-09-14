import numpy as np

import stormpy

import scipy
from scipy.sparse.linalg import factorized, bicgstab, spilu, splu, LinearOperator, gmres

from tqdm import tqdm


class StormSolver:
    def __init__(self, filename, property: str, model_parameters: dict = None, add_dont_care_action=False):
        self.filename = filename
        self.property = property
        self.model_parameters = model_parameters
        self.add_dont_care_action = add_dont_care_action

        if filename.endswith(".drn"):
            mdp, self.parsed_property_ = self.__load_drn_model(self.filename)
        else:
            mdp, self.parsed_property_ = self.__load_prism_model(self.filename)

        # TODO: possibly filter unreachable states
        self.model_ = self.__prepare_model(mdp)
        self.choice_labeling_ = self.model_.choice_labeling  # NOTE: referencing choice_labeling is expensive so only do it once

        self.n_states_ = self.model_.nr_states
        self.n_actions_ = len(self.action_name_to_id_)
        self.action_names_ = [self.action_id_to_name_[i] for i in range(len(self.action_id_to_name_))]

        if len(self.model_.initial_states) > 1:
            raise ValueError("Currently only models with one initial state supported")

        self.initial_state_ = self.model_.initial_states[0]

        self.sparse_trans_probs_, self.feasible_actions_, self.state_action_to_local_action_ = self.__compute_sparse_transitions()

        # Initialize an environment to suppress warnings about inefficient reinitialization of VI operator
        self.environment_ = stormpy.Environment()

    def __load_drn_model(self, filename):
        options = stormpy.DirectEncodingParserOptions()
        options.build_choice_labels = True

        drn_model = stormpy.build_model_from_drn(filename, options)

        first_property = stormpy.parse_properties(self.property)[0]
        return drn_model, first_property

    def __load_prism_model(self, filename):
        prism_program = stormpy.parse_prism_program(filename)
        prism_constants = {}
        mgr = prism_program.expression_manager
        if self.model_parameters:
            for name, value in self.model_parameters.items():
                prism_variable = next(
                    c for c in prism_program.constants if c.name == name
                ).expression_variable
                prism_constants[prism_variable] = mgr.create_integer(value)
        prism_program = prism_program.define_constants(prism_constants)

        options = stormpy.BuilderOptions()
        options.set_build_state_valuations(True)
        options.set_build_all_reward_models(True)
        options.set_build_choice_labels(True)

        mdp = stormpy.build_sparse_model_with_options(prism_program, options)

        first_property = stormpy.parse_properties(self.property, prism_program)[0]
        return mdp, first_property
    
    def __prepare_model(self, mdp):
        model_choice_labeling = mdp.choice_labeling

        action_labels = model_choice_labeling.get_labels()
        action_id_to_name = {i: label for i, label in enumerate(action_labels)}

        for choice_id in range(mdp.nr_choices):
            labels = model_choice_labeling.get_labels_of_choice(choice_id)
            if len(labels) == 0:
                next_id = len(action_id_to_name)
                action_id_to_name[next_id] = "__no_label__"
                print("Added an empty label action")
                break

        self.action_id_to_name_ = action_id_to_name
        self.action_name_to_id_ = {v: k for k, v in self.action_id_to_name_.items()}

        # Make sure the MDP uses state-action rewards
        compatible_reward_models = {}
        for reward_model_name in mdp.reward_models:
            reward_model = mdp.reward_models[reward_model_name]
            if not reward_model.has_state_action_rewards:
                if reward_model.has_state_rewards:
                    state_action_rewards = []
                    for state in mdp.states:
                        state_reward = reward_model.state_rewards[state.id]
                        for _ in state.actions:
                            state_action_rewards.append(state_reward)
                    new_reward_model = stormpy.SparseRewardModel(
                        optional_state_action_reward_vector=state_action_rewards
                    )
                    compatible_reward_models[reward_model_name] = new_reward_model
                else:
                    raise RuntimeError("Could not compute state_action_rewards")
            else:
                compatible_reward_models[reward_model_name] = reward_model

        if self.add_dont_care_action:
            if "__random__" in self.action_name_to_id_:
                raise ValueError("add_dont_care_action supplied but there is already a __random__ action in the MDP")

            # To add a "don't care action" to we add a random action to each state that picks a random available action
            next_id = len(action_id_to_name)
            action_id_to_name[next_id] = "__random__"

            self.action_id_to_name_ = action_id_to_name
            self.action_name_to_id_ = {v: k for k, v in self.action_id_to_name_.items()}
            
            old_state_action_rewards = {reward_name: compatible_reward_models[reward_name].state_action_rewards for reward_name in compatible_reward_models}
            state_action_rewards = {reward_name: [] for reward_name in old_state_action_rewards}
            builder = stormpy.SparseMatrixBuilder(
                rows=0,
                columns=mdp.nr_states,
                entries=0,
                row_groups=mdp.nr_states,
                force_dimensions=True,
                has_custom_row_grouping=True,
            )

            old_trans_matrix = mdp.transition_matrix

            rows = 0
            row_to_choice_label = {}
            for state_id in tqdm(range(mdp.nr_states)):
                builder.new_row_group(rows)

                random_transitions = {}
                random_state_action_rewards = {reward_name: 0.0 for reward_name in state_action_rewards}

                indices_group = old_trans_matrix.get_rows_for_group(state_id)
                n_actions_state = len(indices_group)
                for old_row_id in indices_group:
                    labels = model_choice_labeling.get_labels_of_choice(old_row_id)
                    label = "__no_label__" if len(labels) == 0 else next(iter(labels))
                    action_id = self.action_name_to_id_[label]

                    row = old_trans_matrix.get_row(old_row_id)

                    # copy transitions
                    for entry in row:
                        next_state = entry.column
                        probability = entry.value()
                        builder.add_next_value(rows, next_state, probability)

                        if next_state not in random_transitions:
                            random_transitions[next_state] = probability / n_actions_state
                        else:
                            random_transitions[next_state] += probability / n_actions_state

                    for reward_name in state_action_rewards:
                        state_action_reward = old_state_action_rewards[reward_name][old_row_id]
                        state_action_rewards[reward_name].append(state_action_reward)
                        random_state_action_rewards[reward_name] += state_action_reward / n_actions_state

                    if n_actions_state == 1:
                        row_to_choice_label[rows] = "__random__"
                    else:
                        row_to_choice_label[rows] = action_id_to_name[action_id]
                    rows += 1
                
                if n_actions_state > 1:
                    # Add the actual random action
                    for column in random_transitions:
                        builder.add_next_value(rows, column, random_transitions[column])

                    for reward_name in random_state_action_rewards:
                        state_action_rewards[reward_name].append(random_state_action_rewards[reward_name])

                    row_to_choice_label[rows] = "__random__"
                    rows += 1

            transition_matrix = builder.build()

            choice_labeling = stormpy.storage.ChoiceLabeling(rows)
            for label in self.action_name_to_id_:
                choice_labeling.add_label(label)
            for row, label in row_to_choice_label.items():
                choice_labeling.add_label_to_choice(label, row)

            reward_models = {
                name: stormpy.SparseRewardModel(
                    optional_state_action_reward_vector=state_action_rewards[name]
                )
                for name in state_action_rewards
            }

            components = stormpy.SparseModelComponents(
                transition_matrix=transition_matrix,
                state_labeling=mdp.labeling,
                reward_models=reward_models,
            )
            components.choice_labeling = choice_labeling

            if mdp.has_state_valuations():
                components.state_valuations = mdp.state_valuations
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=mdp.transition_matrix,
                state_labeling=mdp.labeling,
                reward_models=compatible_reward_models,
            )
            components.choice_labeling = model_choice_labeling

            if mdp.has_state_valuations():
                components.state_valuations = mdp.state_valuations
        mdp = stormpy.storage.SparseMdp(components)

        return mdp


    def __create_scheduler(self):
        trivial_property = stormpy.parse_properties("Pmax=? [ F true ]")[0]
        res = stormpy.model_checking(
            self.model_,
            trivial_property,
            environment=self.environment_,
            extract_scheduler=True,
        )
        return res.scheduler

    def __policy_to_scheduler(self, policy: np.ndarray):
        # NOTE: stormpy currently does not allow for custom scheduler (policy) creation
        # so we solve a trivial property once and overwrite the scheduler when we want
        # to define a custom scheduler
        scheduler = self.__create_scheduler()

        for state_id in range(self.n_states_):
            state_action_id = self.state_action_to_local_action_[state_id, policy[state_id]]
            choice = stormpy.SchedulerChoice(state_action_id)
            scheduler.set_choice(choice, state_id)
        return scheduler

    def __action_to_index(self, action):
        if len(action.labels) == 0:
            action_label = "__no_label__"
        else:
            action_label = next(iter(action.labels))
        return self.action_name_to_id_[action_label]

    def __compute_sparse_transitions(self):
        action_transition_probas = []
        for _ in range(self.n_actions_):
            action_transition_probas.append(({}))
    
        feasible_actions = []
        state_action_to_local_action = {}
        trans_matrix = self.model_.transition_matrix
        for state_id in tqdm(range(self.n_states_)):
            indices_group = trans_matrix.get_rows_for_group(state_id)
            actions_state = set()

            for local_action_id, row_id in enumerate(indices_group):
                labels = self.choice_labeling_.get_labels_of_choice(row_id)
                label = "__no_label__" if len(labels) == 0 else next(iter(labels))
                action_id = self.action_name_to_id_[label]
                transition_probas = action_transition_probas[action_id]

                actions_state.add(action_id)
                state_action_to_local_action[state_id, action_id] = local_action_id

                # copy transitions
                for entry in trans_matrix.get_row(row_id):
                    next_state = entry.column
                    probability = entry.value()

                    key = (state_id, next_state)
                    if key in transition_probas:
                        transition_probas[key] += probability
                    else:
                        transition_probas[key] = probability
            
            feasible_actions.append(actions_state)

        sparse_trans_probs = []
        for transition_probas in action_transition_probas:
            row_ind = []
            col_ind = []
            data = []
            for (state, next_state), proba in transition_probas.items():
                row_ind.append(state)
                col_ind.append(next_state)
                data.append(proba)
            sparse_matrix = scipy.sparse.csr_matrix(
                (data, (row_ind, col_ind)), shape=(self.n_states_, self.n_states_)
            )
            sparse_trans_probs.append(sparse_matrix)
        return sparse_trans_probs, feasible_actions, state_action_to_local_action

    def get_observations(self):
        if self.model_.has_state_valuations():
            self.feature_names_ = [
                predicate.strip().split("=")[0]
                for predicate in self.model_.states[0].valuations[1:-1].split("&")
            ]
            observations = np.empty((self.n_states_, len(self.feature_names_)))
            for state in self.model_.states:
                state_obs = []
                for predicate in state.valuations[1:-1].split("&"):
                    predicate = predicate.strip()
                    if "=" in predicate:
                        state_obs.append(float(predicate.split("=")[1]))
                    else:
                        if predicate.startswith("!"):
                            state_obs.append(0.0)
                        else:
                            state_obs.append(1.0)
                observations[state.id] = state_obs
            return observations
        elif self.filename.endswith(".drn"):
            with open(self.filename, "r") as file:
                obs_dicts = []
                lines = file.readlines()
                for line, next_line in zip(lines[:-1], lines[1:]):
                    if line.startswith("state"):
                        if not next_line.startswith("//"):
                            raise ValueError(f"Could not find state valuation comment for state: {line} in {next_line}")
                        
                        predicates_and_vals = [term.strip() for term in next_line[2:].strip("\n[]").split("&")]
                        obs_dict = {}
                        for predicate_and_val in predicates_and_vals:
                            if "=" in predicate_and_val:
                                predicate, value = predicate_and_val.split("=")
                                obs_dict[predicate.strip()] = float(value)
                            else:
                                stripped = predicate_and_val.strip()
                                if stripped.startswith("!"):
                                    obs_dict[stripped[1:]] = 0.0
                                else:
                                    obs_dict[stripped] = 1.0
                        obs_dicts.append(obs_dict)
                assert len(obs_dicts) == self.n_states_

                self.feature_names_ = list(obs_dicts[0].keys())
                observations = np.empty((self.n_states_, len(self.feature_names_)))
                for state_i, obs_dict in enumerate(obs_dicts):
                    assert len(obs_dict) == len(self.feature_names_), f"unable to parse all features for state {state_i}"

                    for predicate, value in obs_dict.items():
                        observations[state_i, self.feature_names_.index(predicate)] = value
                return observations
        else:
            raise ValueError("Cannot extract observations, need DRN file or model built with state valuations")

    def __get_model(self, fixed_state_actions=None):
        # TODO: optimize this function
        if fixed_state_actions is None:
            return self.model_

        if "__random__" in self.action_name_to_id_:
            random_action_id = self.action_name_to_id_["__random__"]
        else:
            random_action_id = None
        
        choices_to_disable = set()
        for state_i in fixed_state_actions:
            if len(self.model_.states[state_i].actions) == 1:
                continue
            for action in self.model_.states[state_i].actions:
                action_id = self.__action_to_index(action)
                
                if action_id not in fixed_state_actions[state_i]:
                    if action_id == random_action_id:
                        continue

                    choice_index = self.model_.get_choice_index(state_i, action.id)
                    choices_to_disable.add(choice_index)

        keep_choices = stormpy.BitVector(self.model_.nr_choices, True)
        for choice_to_disable in choices_to_disable:
            keep_choices.set(choice_to_disable, False)
        keep_states = stormpy.BitVector(self.n_states_, True)

        transition_matrix = self.model_.transition_matrix.submatrix(
            keep_choices, keep_states, use_groups=False
        )

        choice_labeling = stormpy.storage.ChoiceLabeling(
            self.model_.nr_choices - len(choices_to_disable)
        )
        for label in self.choice_labeling_.get_labels():
            choice_labeling.add_label(label)

        new_choice_i = 0
        for choice in range(self.model_.nr_choices):
            if choice not in choices_to_disable:
                for label in self.choice_labeling_.get_labels_of_choice(choice):
                    choice_labeling.add_label_to_choice(label, new_choice_i)
                new_choice_i += 1

        reward_models = {}
        for name in self.model_.reward_models:
            old_rewards = self.model_.reward_models[name].state_action_rewards
            rewards = [
                rew for i, rew in enumerate(old_rewards) if i not in choices_to_disable
            ]
            reward_models[name] = stormpy.SparseRewardModel(
                optional_state_action_reward_vector=rewards
            )

        components = stormpy.storage.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=self.model_.labeling,
            reward_models=reward_models,
        )
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseMdp(components)
        return mdp

    def solve(self, fixed_state_actions=None, initial_values=None, initial_policy=None):
        model = self.__get_model(fixed_state_actions)
        result = stormpy.model_checking(
            model,
            self.parsed_property_,
            environment=self.environment_,
            extract_scheduler=True,
        )
        values = np.array(result.get_values())

        if not result.has_scheduler:
            raise RuntimeError("Could not find scheduler")

        scheduler = result.scheduler
        policy = np.zeros(self.n_states_, dtype=int)
        for state_id in range(self.n_states_):
            state_action_id = scheduler.get_choice(state_id).get_deterministic_choice()

            for action in model.states[state_id].actions:
                if action.id == state_action_id:
                    action_id = self.__action_to_index(action)
                    break
            else:
                raise RuntimeError("Could not find correct action")

            policy[state_id] = action_id

        # TODO: eventually support minimization in branch and bound?
        if "min=" in self.property:
            values = -values

        objective = values[self.initial_state_]

        return values, policy, objective

    def allowed_actions(self, state_id):
        # If we have a "don't care action" we can just pick any action
        if "__random__" in self.action_name_to_id_:
            return set(range(self.n_actions_))
        
        return self.feasible_actions(state_id)
    
    def feasible_actions(self, state_id):
        return self.feasible_actions_[state_id]

    def values_to_q_values(
        self,
        values: np.ndarray,
        fixed_state_actions=None,
    ):
        q_values = np.zeros((self.n_states_, self.n_actions_))
        for action in range(self.n_actions_):
            q_values[:, action] = self.sparse_trans_probs_[action].dot(values)
            states_defined_sum = np.asarray(
                self.sparse_trans_probs_[action].sum(axis=1)
            ).squeeze()
            q_values[states_defined_sum == 0, action] -= 10e10
        
        if "__random__" in self.action_name_to_id_:
            random_action_id = self.action_name_to_id_["__random__"]
            for action in range(self.n_actions_):
                if action != random_action_id:
                    states_defined_sum = np.asarray(
                        self.sparse_trans_probs_[action].sum(axis=1)
                    ).squeeze()
                    q_values[states_defined_sum == 0, action] = q_values[states_defined_sum == 0, random_action_id]

        return q_values
    
    def __patch_dont_care_policy(self, policy: np.ndarray):
        patched_policy = policy.copy()
        for state_id, action in enumerate(patched_policy):
            if action not in self.feasible_actions(state_id):
                patched_policy[state_id] = self.action_name_to_id_["__random__"]
        return patched_policy

    def compute_discounted_state_dist(self, policy: np.ndarray):
        if "__random__" in self.action_name_to_id_:
            policy = self.__patch_dont_care_policy(policy)

        blocks = []
        row_order_parts = []
        for a in range(self.n_actions_):
            idx = np.nonzero(policy == a)[0]
            if idx.size:
                # sparse_trans_probs[a][idx] -> block with rows in the order of idx
                blocks.append(
                    self.sparse_trans_probs_[a][idx]
                )  # slicing a CSR by row gives CSR
                row_order_parts.append(idx)

        if not blocks:
            # no states? return empty square csr
            P_pi = scipy.sparse.csr_matrix((self.n_states_, self.n_states_))
        else:
            P_block = scipy.sparse.vstack(blocks, format="csr")
            row_order = np.concatenate(
                row_order_parts
            )  # maps P_block row -> original state index
            # perm such that row_order[perm] == np.arange(n_states)
            perm = np.argsort(row_order)
            P_pi = P_block[perm]  # reorder rows so row i corresponds to state i
            assert P_pi.shape == (self.n_states_, self.n_states_)

        A = (scipy.sparse.identity(self.n_states_, format="csc")
            - 0.999 * P_pi.T.tocsc())
        
        initial_state_probs = np.zeros(self.n_states_)
        initial_state_probs[self.initial_state_] = 1.0

        lu = splu(A)
        result = lu.solve(initial_state_probs)

        # ilu = spilu(A, fill_factor=10, drop_tol=1e-4)
        # M = LinearOperator(shape=A.shape, matvec=ilu.solve)
        # result, info = gmres(A, initial_state_probs, M=M, atol=1e-8, maxiter=500, restart=50)
        # if info != 0:
        #     print(f"gmres did not converge: info={info}")

        return result

    def evaluate_policy(
        self,
        policy: np.ndarray,
    ):
        if "__random__" in self.action_name_to_id_:
            policy = self.__patch_dont_care_policy(policy)

        assert len(policy) == self.n_states_
        
        try:
            scheduler = self.__policy_to_scheduler(policy)
        except ValueError:
            return -np.inf

        dtmc = self.model_.apply_scheduler(scheduler, drop_unreachable_states=False)
        result = stormpy.model_checking(
            dtmc, self.parsed_property_, environment=self.environment_
        )
        values = result.get_values()

        if "min=" in self.property:
            return -values[self.initial_state_]
        else:
            return values[self.initial_state_]
