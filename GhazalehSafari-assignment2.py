#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ghazaleh Safari


Integrated International Master- and PhD program in Mathematics 
"""

import numpy as np
import networkx as nx

# Agent class representing individual agents in the simulation
class Agent():
    def __init__(self, S, id_number, choice_function_exponent):
        """
        Constructor method.

        Parameters
        ----------
        S : Simulation object
            The simulation the agent belongs to.
        id_number : int
            Unique ID number of the agent.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.

        Returns
        -------
        None.

        """
        self.id_number = id_number
        self.Simulation = S
        self.technology = None
        self.choice_function_exponent = choice_function_exponent

    def choose(self):
        """
        Method for an agent to choose a technology.

        Returns
        -------
        old_tech : int or None
            The agent's previous technology choice.
        new_tech : int or None
            The agent's new technology choice.

        """
        neighbors = self.get_neighbors()
        tech_list = self.Simulation.get_technologies_list()
        tech_frequency = {tech: 0 for tech in tech_list}
        for A in neighbors:
            tech = A.get_technology()
            if tech is not None:
                tech_frequency[tech] += 1
        """ Compute choice probabilities based on the distribution in the 
            immediate neighborhood. The form of the transformation may tend to 
            the technology used by the majority (if self.choice_function_exponent > 1)
            or overrepresent to those used by the minority (if 
            self.choice_function_exponent < 1)"""
        tech_probability = [tech_frequency[tech] ** self.choice_function_exponent for tech in tech_list]
        
        if np.sum(tech_probability) > 0:
            """ Select and adopt a technology"""
            tech_probability = np.asarray(tech_probability) / np.sum(tech_probability)
            old_tech = self.technology
            self.technology = np.random.choice(tech_list, p=tech_probability)
            """ Report the change back"""
            return old_tech, self.technology
        else:
            """ Report that no change was possible"""
            return None, None
        

    def get_technology(self):
        """
        Getter method for the technology the agent uses.

        Returns
        -------
        int
            Current technology. The technologies are characterized as ints.

        """
        return self.technology
        
    def set_technology(self, tech):
        """
        Setter method for the technology the agent uses.

        Parameters
        ----------
        tech : int
            New technology the agent should adopt. The technologies are 
            characterized as ints.

        Returns
        -------
        None.

        """
        self.technology = tech
        
    def get_neighbors(self):
        """
        Get the neighbors of the agent.

        Returns
        -------
        list
            A list of neighboring agents.

        """
        return [self.Simulation.G.nodes[N]["agent"] for N in nx.neighbors(self.Simulation.G, self.id_number)]

# Simulation class representing the entire simulation process
class Simulation():
    def __init__(self, seed=0, n_agents=1000, n_technologies=3, n_initial_adopters=2,
                 reconsideration_probability=0.2, choice_function_exponent=2,
                 network_type="Erdos-Renyi", t_max=5000):
        """
        Constructor method.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. The default is 0.
        n_agents : int, optional
            Number of agents. The default is 1000.
        n_technologies : int, optional
            Number of technologies. The default is 3.
        n_initial_adopters : int, optional
            Number of initial adopters of each technology. The default is 2.
        reconsideration_probability : float, optional
            Probability for agents that have already chosen to reconsider their 
            choice when given the chance. The default is 0.2.
        choice_function_exponent : numeric, optional
            Exponent of the Generalized Eggenberger-Polya process choice 
            function. Values >1 will lead to winner-take-all dynamics, Values 
            <1 lead to equalization dynamics. The default is 2.
        network_type : str, optional
            Network type. Can be Erdos-Renyi, Barabasi-Albert, or Watts-Strogatz. 
            The default is "Erdos-Renyi".
        t_max : int, optional
            Number of time periods. The default is 5000.

        Returns
        -------
        None.

        """
        self.seed = seed
        self.n_agents = n_agents
        self.t_max = 5000
        self.n_technologies = n_technologies
        self.n_initial_adopters = n_initial_adopters
        self.reconsideration_probability = reconsideration_probability
        self.choice_function_exponent = choice_function_exponent

        """ Prepare technology list"""
        self.technologies_list = list(range(self.n_technologies))
        """ Prepare technology frequency dict. Each technology initialized with
            number zero."""
        self.tech_frequency = {tech: 0 for tech in self.technologies_list}
        
        """ Generate network"""
        if network_type == "Erdos-Renyi":
            self.G = nx.erdos_renyi_graph(n=self.n_agents, p=0.1, seed=self.seed)
        elif network_type == "Barabasi-Albert":
            self.G = nx.barabasi_albert_graph(n=self.n_agents, m=40, seed=self.seed)
        elif network_type == "Watts-Strogatz":
            self.G = nx.connected_watts_strogatz_graph(n=self.n_agents, k=40, p=0.15, seed=self.seed)
        else:
            assert False, "Unknown network type {:s}".format(network_type)

        """ Create agents and place them on the network"""
        self.agents_list = []

        for i in range(self.G.order()):
            A = Agent(self, i, self.choice_function_exponent)
            self.agents_list.append(A)
            self.G.nodes[i]["agent"] = A

        for tech in self.technologies_list:
            n_early_adopters = self.n_technologies * self.n_initial_adopters
            early_adopters = list(np.random.choice(self.agents_list, replace=False, size=n_early_adopters))
            for i in range(self.n_technologies):
                for j in range(self.n_initial_adopters):
                    early_adopters[i * self.n_initial_adopters + j].set_technology(i)
                    self.tech_frequency[i] += 1

        self.history_tech_frequency = {tech: [] for tech in self.technologies_list}

    def get_technologies_list(self):
        """
        Get the list of technologies.

        Returns
        -------
        list
            A list of technology choices.

        """
        return self.technologies_list

    def return_results(self):
        """
        Return the simulation results.

        Returns
        -------
        dict
            A dictionary containing the simulation results.

        """
        return {
            'history_tech_frequency': self.history_tech_frequency,
        }

    def run(self):
        """
        Run the simulation.

        Returns
        -------
        None.

        """
        np.random.seed(self.seed)
        for t in range(self.t_max):
            for A in self.agents_list:
                if np.random.rand() < self.reconsideration_probability:
                    old_tech, new_tech = A.choose()
                    if old_tech is not None:
                        self.tech_frequency[old_tech] -= 1
                else:
                    new_tech = A.get_technology()

                if new_tech is not None:
                    self.tech_frequency[new_tech] += 1

            for tech in self.technologies_list:
                self.history_tech_frequency[tech].append(self.tech_frequency[tech])

""" Main entry point"""  
if __name__ == '__main__':
    np.random.seed(0)# Set the seed value for reproducibility
    S = Simulation(seed=0)
    S.run()
    results = S.return_results()
    
    # Compute the market share of the second-most widely used technology
    tech_frequency = results['history_tech_frequency']
    market_shares = np.array(list(tech_frequency.values()))
    second_market_share = np.sort(market_shares[:, -1])[-2] / S.n_agents
    print("Market share of the second-most widely used technology:", second_market_share)
