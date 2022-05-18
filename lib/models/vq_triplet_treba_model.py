import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal

from lib.models.loss import (compute_label_loss,
                             compute_decoding_loss, compute_contrastive_loss,
                             compute_triplet_loss, compute_quantization_loss)

from lib.models.treba_model import TREBA_model
from lib.models.vq_model import VQEmbedding


class VQTripletTREBA_model(TREBA_model):
    name = 'VQTripletTREBA_model'

    # Required arguments to the model.
    model_args = [
        'state_dim',
        'action_dim',
        'z_dim',
        'h_dim',
        'rnn_dim',
        'num_layers'
    ]
    requires_labels = False
    requires_augmentations = False
    log_metrics = False

    # Default hyperparameters used for training.
    loss_params = {
        "contrastive_temperature": 0.07,
        "contrastive_base_temperature": 0.07,
        "consistency_temperature": 0.1,
        "contrastive_loss_weight": 1000.0,
        "consistency_loss_weight": 1.0,
        "decoding_loss_weight": 0.0,
        "triplet_loss_weight": 1.0,
        "quantization_loss_weight": 1.0
    }

    label_functions = []

    def __init__(self, model_config):
        super().__init__(model_config)

    def _construct_model(self):
        super()._construct_model()

        # num_embeddings controls the number of clusters to use
        num_embeddings = self.config['num_embeddings']

        # high commitment cost means the encoder outputs are very
        # responsive to the codebook vector locations
        commitment_cost = self.config['commitment_cost']

        # Define the codebook module
        self.codebook = VQEmbedding(num_embeddings, self.config['z_dim'])

    def _define_losses(self):
        super()._define_losses()
        self.log.add_loss('triplet')
        self.log.add_loss('quantization')

    def model_params(self):
        params = super().model_params()
        params += list(self.codebook.parameters())

        return params

    def compute_context(self, ctxt_states, ctxt_actions, ctxt_labels_dict):
        """
        This functions computes the average embedding across the provided
        context states
        """
        ctxt_labels = None
        if len(ctxt_labels_dict) > 0:
            ctxt_labels = torch.cat(list(ctxt_labels_dict.values()), dim=-1)

        (ctxt_states, ctxt_actions, ctxt_labels) = (ctxt_states.transpose(1, 0),
                                                    ctxt_actions.transpose(1, 0),
                                                    ctxt_labels.transpose(1, 0))

        ctxt_embeddings = []
        for states, actions, labels in zip(ctxt_states, ctxt_actions, ctxt_labels):
            states = states.transpose(0, 1)
            actions = actions.transpose(0, 1)
            ctxt_embeddings.append(self.encode_mean(states[:-1], actions=actions, labels=labels)[0])

        return torch.mean(torch.stack(ctxt_embeddings), dim=0)

    def find_negatives(self, states, actions, labels_dict):
        """
        This function will find negatives such that they differ
        from the positive by at least num_diff features. A negative
        is considered different from a positive w.r.t. a particular
        feature if it is at least num_bins away from whichever bin
        the positive falls into.
        """
        with torch.no_grad():
            neg_states, neg_actions, neg_labels = [], [], []
            states = states.transpose(0, 1)
            actions = actions.transpose(0, 1)

            labels = None
            if len(labels_dict) > 0:
                labels = torch.cat(list(labels_dict.values()), dim=-1)

            for i, state in enumerate(states):
                feats = torch.split(labels, self.config['num_bins'], dim=1)

                diffs = []
                for feat in feats:
                    un_ohe = torch.where(feat == 1)[1]
                    diff_idxs = torch.where(torch.abs(un_ohe - un_ohe[i]) \
                                            >= self.config['bin_dist'])[0]
                    diff_by_feat = torch.zeros(feat.shape[0])
                    diff_by_feat[diff_idxs] = 1
                    diffs.append(diff_by_feat.reshape(-1, 1))
                diffs = torch.cat(diffs, dim=1)
                diffs = torch.where(torch.sum(diffs, axis=1) > 1)[0]

                rand_neg_idx = diffs[torch.randint(len(diffs), (1,))]

                neg_states.append(states[rand_neg_idx])
                neg_actions.append(actions[rand_neg_idx])
                neg_labels.append(labels[rand_neg_idx])

            neg_states = torch.cat(neg_states, dim=0)
            neg_actions = torch.cat(neg_actions, dim=0)
            neg_labels = torch.cat(neg_labels, dim=0)

            return neg_states.transpose(0, 1), neg_actions.transpose(0, 1), neg_labels

    def forward(
            self, states, actions, labels_dict,
            ctxt_states, ctxt_actions, ctxt_labels_dict,
            embed=False, restart=False, reconstruct=False
    ):
        self.log.reset()

        # Consistency and decoding loss need labels.
        if (self.loss_params['consistency_loss_weight'] > 0 or
                self.loss_params['decoding_loss_weight'] > 0):
            assert len(labels_dict) > 0

        assert actions.size(1) + 1 == states.size(1)  # final state has no corresponding action
        states = states.transpose(0, 1)
        actions = actions.transpose(0, 1)

        labels = None
        if len(labels_dict) > 0:
            labels = torch.cat(list(labels_dict.values()), dim=-1)

        # Pretrain program approximators, if using consistency loss.
        if self.stage == 1 and self.loss_params['consistency_loss_weight'] > 0:
            for lf_idx, lf_name in enumerate(labels_dict):
                lf = self.config['label_functions'][lf_idx]
                lf_labels = labels_dict[lf_name]
                self.log.losses[lf_name] = compute_label_loss(
                    states[:-1],
                    actions,
                    lf_labels,
                    self.label_approx_birnn[lf_idx],
                    self.label_approx_fc[lf_idx],
                    lf.categorical
                )

                # If weighting a particular type of labeling function
                if hasattr(lf, 'weight'):
                    self.log.losses[lf_name] *= lf.weight

                # Compute label loss with approx
                if self.log_metrics:
                    approx_labels = self.label(states[:-1], actions, lf_idx, lf.categorical)
                    assert approx_labels.size() == lf_labels.size()
                    self.log.metrics['{}_approx'.format(lf.name)] = torch.sum(approx_labels * lf_labels)

        # Train TVAE with programs.
        elif self.stage >= 2 or not self.loss_params['consistency_loss_weight'] > 0:
            # Encode
            posterior = self.encode(states[:-1], actions=actions, labels=labels)

            # Compute the contextual embedding
            z_ctxt = self.compute_context(ctxt_states, ctxt_actions, ctxt_labels_dict)
            # Find the negatives satisfying conditions specified in config
            n_states, n_actions, n_labels = self.find_negatives(states, actions, labels_dict)

            # Compute the embedding of the negative
            z_n = self.encode(n_states[:-1], actions=n_actions, labels=n_labels).mean

            # Compute the triplet loss
            self.log.losses['triplet'], _, _ = compute_triplet_loss(
                posterior.mean,
                z_n,
                z_ctxt,
                margin=self.config['margin'],
                loss_weight=self.config['triplet_loss_weight']
            )

            # Quantize
            z_q_x_st, z_q_x = self.codebook.straight_through(posterior.mean)

            # Compute quantization_loss
            self.log.losses['quantization'] = compute_quantization_loss(
                z_q_x,
                posterior.mean,
                self.config['commitment_cost'],
                loss_weight=self.config['quantization_loss_weight']
            )

            # Decode
            self.reset_policy(labels=labels, z=z_q_x_st)

            # Compute reconstruction loss
            for t in range(actions.size(0)):
                action_likelihood = self.decode_action(states[t])
                self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])

                if self.is_recurrent:
                    self.update_hidden(states[t], actions[t])

            # Compute decoding loss.
            if self.loss_params['decoding_loss_weight'] > 0:
                # Compute label loss
                for lf_idx, lf_name in enumerate(labels_dict):
                    lf = self.config['label_functions'][lf_idx]
                    lf_labels = labels_dict[lf_name]
                    self.log.losses["decoded_" + lf_name] = compute_decoding_loss(
                        z_q_x_st,
                        lf_labels,
                        self.label_decoder_fc_decoding[lf_idx],
                        lf.categorical,
                        loss_weight=self.loss_params['decoding_loss_weight']
                    )

                    # If weighting specific features
                    if hasattr(lf, 'weight'):
                        self.log.losses["decoded_" + lf_name] *= lf.weight

            # Generate rollout for consistency loss.
            # Use the posterior to train here.
            if self.loss_params['consistency_loss_weight'] > 0:
                self.reset_policy(
                    labels=labels,
                    z=z_q_x_st,
                    temperature=self.loss_params['consistency_temperature']
                )

                rollout_states, rollout_actions = self.generate_rollout(states, horizon=actions.size(0))

                # Compute label loss
                for lf_idx, lf_name in enumerate(labels_dict):
                    lf = self.config['label_functions'][lf_idx]
                    lf_labels = labels_dict[lf_name]
                    self.log.losses[lf_name + '_consistency'] = compute_label_loss(
                        rollout_states[:-1],
                        rollout_actions,
                        lf_labels,
                        self.label_approx_birnn[lf_idx],
                        self.label_approx_fc[lf_idx],
                        lf.categorical,
                        loss_weight=self.loss_params['consistency_loss_weight']
                    )

                    # If weighting specific features
                    if hasattr(lf, 'weight'):
                        self.log.losses[lf_name + '_consistency'] *= lf.weight

                    # Compute label loss with approx
                    if self.log_metrics:
                        approx_labels = self.label(
                            rollout_states[:-1],
                            rollout_actions,
                            lf_idx,
                            lf.categorical
                        )
                        assert approx_labels.size() == lf_labels.size()
                        self.log.metrics['{}_approx'.format(lf.name)] = torch.sum(approx_labels * lf_labels)

                        # Compute label loss with true LF
                        rollout_lf_labels = lf.label(
                            rollout_states.transpose(0, 1).detach().cpu(),
                            rollout_actions.transpose(0, 1).detach().cpu(),
                            batch=True
                        )
                        assert rollout_lf_labels.size() == lf_labels.size()
                        self.log.metrics['{}_true'.format(lf.name)] = torch.sum(rollout_lf_labels * lf_labels.cpu())

        return self.log
