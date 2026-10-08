#!/usr/bin/env python3
"""
Fishing Derby player — BASELINE version (v1).
Idea: one HMM per species, trained with Baum-Welch on the fish whose species has been revealed.
A fish is classified as the species whose model gives its sequence the highest likelihood.
"""
import random

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
from hmm_model import random_model, log_likelihood, baum_welch_multi

# ---- tunable design parameters ----
N_HIDDEN = 2                       # hidden states per species model
START_STEP = N_STEPS - N_FISH      # first step at which we guess (110: just enough steps left)
SEED = 0


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        self.rng = random.Random(SEED)
        self.obs = [[] for _ in range(N_FISH)]          # observation sequence of every fish
        self.models = [None] * N_SPECIES                # trained HMM per species (None = not seen yet)
        self.labelled = [[] for _ in range(N_SPECIES)]  # fish ids whose species has been revealed
        self.guessed = set()

    def guess(self, step, observations):
        for f in range(N_FISH):
            self.obs[f].append(observations[f])

        if step < START_STEP:
            return None
        candidates = [f for f in range(N_FISH) if f not in self.guessed]
        if not candidates:
            return None

        known = [s for s in range(N_SPECIES) if self.models[s] is not None]
        if not known:
            # nothing learned yet: guess the first fish, arbitrary species
            fish, species = candidates[0], self.rng.randrange(N_SPECIES)
        else:
            # classify every candidate; guess the one whose best model fits best (most confident)
            best = None
            for f in candidates:
                T = len(self.obs[f])
                scores = [(log_likelihood(self.models[s], self.obs[f]) / T, s) for s in known]
                score, s = max(scores)
                if best is None or score > best[0]:
                    best = (score, f, s)
            _, fish, species = best

        self.guessed.add(fish)
        return fish, species

    def reveal(self, correct, fish_id, true_type):
        # add the newly labelled fish and retrain that species' model on all its labelled fish
        self.labelled[true_type].append(fish_id)
        seqs = [self.obs[f] for f in self.labelled[true_type]]
        init = self.models[true_type] or random_model(N_HIDDEN, N_EMISSIONS, self.rng)
        self.models[true_type] = baum_welch_multi(init, seqs, N_EMISSIONS)
