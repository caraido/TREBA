import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseSequentialModel
from lib.distributions import Normal

from lib.models.loss import compute_label_loss, compute_decoding_loss, compute_contrastive_loss

class TREBA_model(BaseSequentialModel):

    name = 'TREBA_model' 
    # Required arguments to the model.
    model_args = ['state_dim', 
                'action_dim', 
                'z_dim', 
                'h_dim', 
                'rnn_dim', 
                'num_layers']
    requires_labels = False
    requires_augmentations = False
    
    log_metrics = False

    # Default hyperparameters used for training.
    loss_params = {"contrastive_temperature": 0.07,
                        "contrastive_base_temperature" :0.07,
                        "consistency_temperature" : 0.1,
                        "contrastive_loss_weight" : 1000.0,
                        "consistency_loss_weight" : 1.0,
                        "decoding_loss_weight" : 0.0}

    label_functions = []

    def __init__(self, model_config):
        super().__init__(model_config)

    def _construct_model(self):
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        z_dim = self.config['z_dim']
        h_dim = self.config['h_dim']
        enc_rnn_dim = self.config['rnn_dim']
        dec_rnn_dim = self.config['rnn_dim'] if self.is_recurrent else 0
        label_rnn_dim = self.config['rnn_dim']
        num_layers = self.config['num_layers']

        # QUANTIZATION STUFF
        num_embeddings = self.config['num_embeddings']
        commitment_cost = self.config['commitment_cost']

        for param_key in self.loss_params.keys():
            if param_key in self.config.keys():
                self.loss_params[param_key] = self.config[param_key]

        # Define models used in TREBA.
        self.enc_birnn = nn.GRU(state_dim+action_dim, enc_rnn_dim, 
            num_layers=num_layers, bidirectional=True)

        # If label functions are defined in config.
        if 'label_functions' in self.config.keys():
            self.label_functions = self.config['label_functions']
            label_dim = self.config['label_dim']
        else:
            label_dim = 0

        # Define TVAE Encoder and Decoder.
        self.enc_fc = nn.Sequential(
            nn.Linear(2*enc_rnn_dim+label_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Linear(h_dim, z_dim)

        self.dec_action_fc = nn.Sequential(
            nn.Linear(state_dim+z_dim+dec_rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_action_mean = nn.Linear(h_dim, action_dim)
        self.dec_action_logvar = nn.Linear(h_dim, action_dim)

        # Whether the trajectory decoder is recurrent.
        if self.is_recurrent:
            self.dec_rnn = nn.GRU(state_dim+action_dim, dec_rnn_dim, num_layers=num_layers)

        # Define Models and Decoders for loss functions using labels.
        if len(self.label_functions) > 0:
            # Consistency loss of labels from programmed labeling functions.
            if self.loss_params['consistency_loss_weight'] > 0:

                # Label approximator.
                self.label_approx_birnn = nn.ModuleList([
                    nn.GRU(state_dim+action_dim, label_rnn_dim, 
                        num_layers=num_layers, bidirectional=True) for lf in self.label_functions])

                if 'num_bins' in self.config:
                    self.label_approx_fc = nn.ModuleList([nn.Sequential(
                        nn.Linear(2 * label_rnn_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, self.config['num_bins'])) for lf in self.label_functions])
                else:
                    self.label_approx_fc = nn.ModuleList([nn.Sequential(
                        nn.Linear(2 * label_rnn_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, h_dim),
                        nn.ReLU(),
                        nn.Linear(h_dim, lf.output_dim)) for lf in self.label_functions])

            # Contrastive loss of labels programmatically supervised by the functions.
            if self.loss_params['contrastive_loss_weight'] > 0:

                self.label_decoder_fc_contrastive = nn.ModuleList([nn.Sequential(
                    nn.Linear(z_dim, z_dim),
                    nn.ReLU(),            
                    nn.Linear(z_dim, z_dim)) for lf in self.label_functions])        

            # Decoding loss.
            if self.loss_params['decoding_loss_weight'] > 0:
                self.label_decoder_fc_decoding = nn.ModuleList([nn.Sequential(
                    nn.Linear(z_dim, z_dim),
                    nn.ReLU(),            
                    nn.Linear(z_dim, lf.output_dim)) for lf in self.label_functions])        
        else:
            # If we don't have labels, we can train using unsupervised contrastive loss.
            if self.loss_params['contrastive_loss_weight'] > 0:
                self.label_decoder_fc_contrastive = nn.Sequential(
                                        nn.Linear(z_dim, z_dim),
                                        nn.ReLU(),            
                                        nn.Linear(z_dim, z_dim))               


        # QUANTIZATION STUFF
        self.codebook = torch.nn.Embedding(num_embeddings, z_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.codebook_usage = torch.zeros(num_embeddings)

    def update_usage(self, min_encoding_indices):
        if self.codebook_usage.device != min_encoding_indices.device:
            self.codebook_usage = self.codebook_usage.to(min_encoding_indices.device)
        unq, counts = torch.unique(min_encoding_indices, return_counts=True)
        self.codebook_usage[unq] += counts

    def _define_losses(self):
        self.log.add_loss('kl_div')
        self.log.add_loss('nll')
        self.log.add_metric('kl_div_true')

        # Handle labeling function cases.
        if len(self.label_functions) > 0:

            for lf in self.config['label_functions']:
                if self.config['decoding_loss_weight'] > 0:
                    self.log.add_loss('decoded_' + lf.name)

                if self.loss_params['consistency_loss_weight'] > 0:

                    self.log.add_loss(lf.name + '_consistency')
                    self.log.add_metric('{}_approx'.format(lf.name))
                    self.log.add_metric('{}_true'.format(lf.name))

        # Handle augmentation cases.
        if 'augmentations' in self.config.keys():
            for aug in self.config['augmentations']:
                self.log.add_loss('{}_kl_div'.format(aug.name))
                self.log.add_loss('{}_nll'.format(aug.name))
                self.log.add_metric('{}_kl_div_true'.format(aug.name))

                for lf in self.label_functions:
                    if self.loss_params['decoding_loss_weight'] > 0:
                        self.log.add_loss(aug.name + '_decoded_' + lf.name)      
      
        # Handle cases of unsupervised or programmatically supervised contrastive loss.  
        if self.loss_params['contrastive_loss_weight'] > 0:
            if 'augmentations' in self.config.keys():
                for aug in self.config['augmentations']:

                    if len(self.label_functions) > 0:
                        for lf in self.label_functions:
                            self.log.add_loss(aug.name + '_contrastive_' + lf.name)
                    else:
                        self.log.add_loss(aug.name + '_contrastive')                        
            else:
                if len(self.label_functions) > 0:
                    self.log.add_loss('contrastive_' + lf.name)            
                else:
                    self.log.add_loss('contrastive')


    def model_params(self):
        params = list(self.enc_birnn.parameters()) + list(self.enc_mean.parameters()) + list(self.enc_logvar.parameters()) + \
            list(self.dec_action_fc.parameters()) + list(self.dec_action_mean.parameters())
    
        params += list(self.enc_fc.parameters()) 
        params += list(self.dec_action_logvar.parameters())

        if self.loss_params['contrastive_loss_weight'] > 0:
            params += list(self.label_decoder_fc_contrastive.parameters())

        if len(self.label_functions) > 0 and self.loss_params['decoding_loss_weight'] > 0:
            params += list(self.label_decoder_fc_decoding.parameters())

        if self.is_recurrent:
            params += list(self.dec_rnn.parameters())

        return params

    def label_approx_params(self):
        return list(self.label_approx_birnn.parameters()) + list(self.label_approx_fc.parameters())

    def init_optimizer(self, lr):
        self.model_optimizer = torch.optim.Adam(self.model_params(), lr=lr)

        if len(self.label_functions) > 0 and self.loss_params['consistency_loss_weight'] > 0:
            self.label_approx_optimizer = torch.optim.Adam(self.label_approx_params(), lr=lr)

    def optimize(self, losses):
        assert isinstance(losses, dict)

        # Optimize program approximators.
        # Only need to do this if we are using consistency loss.
        if self.stage == 1 and self.loss_params['consistency_loss_weight'] > 0:
            self.label_approx_optimizer.zero_grad()
            label_preds = [ value for key,value in losses.items() if 'LF' in key ]
            label_approx_loss = sum(label_preds)
            label_approx_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.label_approx_params(), 10)
            self.label_approx_optimizer.step()

        if self.stage >= 2 or not self.loss_params['consistency_loss_weight'] > 0:
            self.model_optimizer.zero_grad()
            model_losses = [ value for key,value in losses.items()]
            model_loss = sum(model_losses)
            model_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.model_params(), 10)
            self.model_optimizer.step()

    def label(self, states, actions, lf_idx, categorical):
        assert states.size(0) == actions.size(0)
        hiddens, _ = self.label_approx_birnn[lf_idx](torch.cat([states, actions], dim=-1))
        avg_hiddens = torch.mean(hiddens, dim=0)
        approx_out = self.label_approx_fc[lf_idx](avg_hiddens)

        approx_out = F.log_softmax(approx_out, dim=-1)
        label_class = torch.argmax(approx_out, dim=1)
        approx_labels = self._one_hot_encode(label_class.long(), approx_out.shape[-1])
        return approx_labels

    def _one_hot_encode(self, labels, dim):
        dims = [labels.size(i) for i in range(len(labels.size()))]
        dims.append(dim)
        label_ohe = torch.zeros(dims).to(labels.device)
        label_ohe.scatter_(-1, labels.unsqueeze(-1), 1)
        return label_ohe

    def compute_context(self, ctxt_states, ctxt_actions, ctxt_labels_dict):
        ctxt_labels = None
        if len(ctxt_labels_dict) > 0:
                ctxt_labels = torch.cat(list(ctxt_labels_dict.values()), dim=-1)
       
        (ctxt_states, ctxt_actions, ctxt_labels) = (ctxt_states.transpose(1,0), 
                                                    ctxt_actions.transpose(1,0), 
                                                    ctxt_labels.transpose(1,0))
 
        ctxt_embeddings = []
        for states, actions, labels in zip(ctxt_states, ctxt_actions, ctxt_labels):

            states = states.transpose(0,1)
            actions = actions.transpose(0,1)
          
            ctxt_embeddings.append(self.encode_mean(states[:-1], actions = actions, labels=labels)[0])
        
        return torch.mean(torch.stack(ctxt_embeddings), dim=0)

    def find_negatives(self, states, actions, labels_dict):
        with torch.no_grad():
            neg_states, neg_actions, neg_labels = [], [], []
            states = states.transpose(0,1)
            actions = actions.transpose(0,1)

            labels = None
            if len(labels_dict) > 0:
                    labels = torch.cat(list(labels_dict.values()), dim=-1)

            for i, state in enumerate(states):
                feats = torch.split(labels, 5, dim=1)

                diffs = []
                for feat in feats:

                    un_ohe = torch.where(feat == 1)[1]
                    diff_idxs = torch.where(torch.abs(un_ohe - un_ohe[i]) > 1)[0]
                    diff_by_feat = torch.zeros(feat.shape[0])
                    diff_by_feat[diff_idxs] = 1
                    diffs.append(diff_by_feat.reshape(-1,1))
                diffs = torch.cat(diffs, dim=1)

                diffs = torch.where(torch.sum(diffs, axis=1) > 1)[0]
                rand_neg_idx = diffs[torch.randint(len(diffs), (1,))]

                neg_states.append(states[rand_neg_idx])
                neg_actions.append(actions[rand_neg_idx])
                neg_labels.append(labels[rand_neg_idx])

            neg_states = torch.cat(neg_states, dim=0)
            neg_actions = torch.cat(neg_actions, dim=0)
            neg_labels = torch.cat(neg_labels, dim=0)

            return neg_states.transpose(0,1), neg_actions.transpose(0,1), neg_labels

    def compute_triplet_loss(self, pos, neg, ctxt):
        p_to_a = torch.linalg.norm(pos - ctxt, axis=1)
        n_to_a = torch.linalg.norm(neg - ctxt, axis=1)
        margin_vec = torch.ones(p_to_a.shape).to(p_to_a.device) * self.config['margin']
        zeros = torch.zeros(p_to_a.shape).to(p_to_a.device)
        triplet_loss = torch.max(p_to_a - n_to_a + margin_vec, zeros).sum()

        return triplet_loss, p_to_a, n_to_a

    def quantize_encode(self, pos, return_idx=False):
            #### first compute the distances from the embedding to each of the codebook vectors
            distances = torch.sum(pos ** 2, dim=1, keepdim=True) + \
                        torch.sum(self.codebook.weight ** 2, dim=1) - \
                        2 * torch.matmul(pos, self.codebook.weight.t())
    
            #### now we need to find the closest codebook vector
            min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0],
                self.config['num_embeddings']).to(pos.device)
            min_encodings.scatter_(1, min_encoding_indices, 1)
    
            #### Mask off the quantized vector
            z_q = torch.matmul(min_encodings, self.codebook.weight).view(pos.shape)

            self.log.losses['quantization'] = torch.mean((z_q.detach() - pos)**2) + \
                                                   self.config['commitment_cost'] * torch.mean((z_q - pos.detach())**2)

            self.update_usage(min_encoding_indices)
 
            return z_q, min_encoding_indices


    def check_usage(self, z_e):
        pmf = (self.codebook_usage / torch.sum(self.codebook_usage))
        dead_code_idxs = torch.where(pmf < self.config['restart_threshold'])
        rand_enc = torch.randperm(z_e.shape[0])[:dead_code_idxs[0].shape[0]]

        with torch.no_grad():
            self.codebook.weight[dead_code_idxs] = z_e[rand_enc]

    def forward(self, states, actions, labels_dict, ctxt_states, ctxt_actions, ctxt_labels_dict, embed=False, restart=False):
        self.log.reset()

        # Consistency and decoding loss need labels.
        if (self.loss_params['consistency_loss_weight'] > 0 or 
            self.loss_params['decoding_loss_weight'] > 0):
            assert len(labels_dict) > 0

        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        labels = None
        if len(labels_dict) > 0:
            labels = torch.cat(list(labels_dict.values()), dim=-1)

        # Pretrain program approximators, if using consistency loss.
        if self.stage == 1 and self.loss_params['consistency_loss_weight'] > 0:
            for lf_idx, lf_name in enumerate(labels_dict):
                lf = self.config['label_functions'][lf_idx]
                lf_labels = labels_dict[lf_name]
                self.log.losses[lf_name] = compute_label_loss(states[:-1], actions, lf_labels,
                                                             self.label_approx_birnn[lf_idx],
                                                             self.label_approx_fc[lf_idx], 
                                                             lf.categorical)

                # Compute label loss with approx
                if self.log_metrics:
                    approx_labels = self.label(states[:-1], actions, lf_idx, lf.categorical)
                    assert approx_labels.size() == lf_labels.size()
                    self.log.metrics['{}_approx'.format(lf.name)] = torch.sum(approx_labels*lf_labels)

        # Train TVAE with programs.
        elif self.stage >= 2 or not self.loss_params['consistency_loss_weight'] > 0:
            # Encode
            posterior = self.encode(states[:-1], actions=actions, labels=labels)

            # -=-= Compute triplet loss =-=-
            pos = posterior.mean
            
            if not embed:
                ctxt = self.compute_context(ctxt_states, ctxt_actions, ctxt_labels_dict)
                neg_states, neg_actions, neg_labels = self.find_negatives(states, actions, labels_dict)
                neg = self.encode(neg_states[:-1], actions=neg_actions, labels=neg_labels).mean
                self.log.losses['triplet'], p_to_a, n_to_a = self.compute_triplet_loss(pos, neg, ctxt)
            # -=-= End compute triplet loss =-=-

            # -=-= Start compute quantization =-=-
            z_q, enc_idx = self.quantize_encode(pos)
            if restart:
                self.check_usage(pos)
            # -=-= End compute quantization =-=-

            # Decode
            self.reset_policy(labels=labels, z=z_q)

            for t in range(actions.size(0)):
                action_likelihood = self.decode_action(states[t])
                self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])

                if self.is_recurrent:
                    self.update_hidden(states[t], actions[t])

            # Add decoding loss.
            if self.loss_params['decoding_loss_weight'] > 0: 
                # Compute label loss
                for lf_idx, lf_name in enumerate(labels_dict):
                    lf = self.config['label_functions'][lf_idx]
                    lf_labels = labels_dict[lf_name]
                    self.log.losses["decoded_" + lf_name] = compute_decoding_loss(z_q, 
                                                    lf_labels, self.label_decoder_fc_decoding[lf_idx], 
                                                    lf.categorical, loss_weight = self.loss_params['decoding_loss_weight'])

            # Generate rollout for consistency loss.
            # Use the posterior to train here.
            if self.loss_params['consistency_loss_weight'] > 0:
                self.reset_policy(labels=labels, z=z_q, 
                        temperature = self.loss_params['consistency_temperature'])

                rollout_states, rollout_actions = self.generate_rollout(states, horizon=actions.size(0))

                # Compute label loss
                for lf_idx, lf_name in enumerate(labels_dict):
                    lf = self.config['label_functions'][lf_idx]
                    lf_labels = labels_dict[lf_name]
                    self.log.losses[lf_name + '_consistency'] = compute_label_loss(rollout_states[:-1], 
                                                            rollout_actions, lf_labels,
                                                            self.label_approx_birnn[lf_idx],
                                                            self.label_approx_fc[lf_idx], 
                                                            lf.categorical, 
                                                            loss_weight = self.loss_params['consistency_loss_weight'])
                    
                    # Compute label loss with approx
                    if self.log_metrics:
                        approx_labels = self.label(rollout_states[:-1], rollout_actions, lf_idx, lf.categorical)
                        assert approx_labels.size() == lf_labels.size()
                        self.log.metrics['{}_approx'.format(lf.name)] = torch.sum(approx_labels*lf_labels)

                        # Compute label loss with true LF    
                        rollout_lf_labels = lf.label(rollout_states.transpose(0,1).detach().cpu(),
                            rollout_actions.transpose(0,1).detach().cpu(), batch=True)
                        assert rollout_lf_labels.size() == lf_labels.size()
                        self.log.metrics['{}_true'.format(lf.name)] = torch.sum(rollout_lf_labels*lf_labels.cpu())

            # If augmentations are provided, additionally train with those.
            if 'augmentations' in self.config.keys():
                for aug in self.config['augmentations']:

                    augmented_states, augmented_actions = aug.augment(states.transpose(0,1), 
                                                    actions.transpose(0,1),
                                                    batch=True)

                    augmented_states = augmented_states.transpose(0,1)
                    augmented_actions = augmented_actions.transpose(0,1)                    
                    aug_posterior = self.encode(augmented_states[:-1], actions=augmented_actions, 
                                        labels=labels)

                    kld = Normal.kl_divergence(aug_posterior, free_bits=0.0).detach()
                    self.log.metrics['{}_kl_div_true'.format(aug.name)] = torch.sum(kld)

                    kld = Normal.kl_divergence(aug_posterior, free_bits=1/self.config['z_dim'])
                    self.log.losses['{}_kl_div'.format(aug.name)] = torch.sum(kld)

                    # Decode
                    self.reset_policy(labels=labels, z=aug_posterior.sample())

                    for t in range(actions.size(0)):
                        action_likelihood = self.decode_action(augmented_states[t])
                        self.log.losses['{}_nll'.format(aug.name)] -= action_likelihood.log_prob(augmented_actions[t])

                        if self.is_recurrent:
                            self.update_hidden(augmented_states[t], augmented_actions[t])                    

                    for lf_idx, lf_name in enumerate(labels_dict):
                        
                        # Train contrastive loss with augmentations and programs.
                        lf_labels = labels_dict[lf_name]
                        if self.loss_params['contrastive_loss_weight'] > 0: 

                            self.log.losses[aug.name + "_contrastive_" + lf_name] = compute_contrastive_loss(posterior.mean, aug_posterior.mean, 
                                self.label_decoder_fc_contrastive[lf_idx], labels = lf_labels, 
                                temperature = self.loss_params['contrastive_temperature'],
                                base_temperature = self.loss_params['contrastive_base_temperature'],
                                loss_weight = self.loss_params['contrastive_loss_weight'])

                        if self.loss_params['decoding_loss_weight'] > 0: 
                            self.log.losses[aug.name + "_decoded_" + lf_name] = compute_decoding_loss(aug_posterior.mean, 
                                                                lf_labels, self.label_decoder_fc_decoding[lf_idx], lf.categorical,
                                                                loss_weight = self.loss_params['decoding_loss_weight'])

                    if len(labels_dict) == 0 and self.loss_params['contrastive_loss_weight'] > 0:
                        # Train with unsupervised contrastive loss.
                        self.log.losses[aug.name + '_contrastive'] = compute_contrastive_loss(posterior.mean, aug_posterior.mean,
                                                                                self.label_decoder_fc_contrastive,
                                                                                temperature = self.loss_params['contrastive_temperature'],
                                                                                base_temperature = self.loss_params['contrastive_base_temperature'],
                                                                                loss_weight = self.loss_params['contrastive_loss_weight'])


            # Add contrastive loss for cases where there are no augmentations.
            if (('augmentations' not in self.config.keys() or len(self.config['augmentations']) == 0) 
                and self.loss_params['contrastive_loss_weight'] > 0): 

                if  len(labels_dict) == 0:
                    self.log.losses['contrastive'] = compute_contrastive_loss(posterior.mean, posterior.mean,
                                                                            self.label_decoder_fc_contrastive,
                                                                            temperature = self.loss_params['contrastive_temperature'],
                                                                            base_temperature = self.loss_params['contrastive_base_temperature'],
                                                                            loss_weight = self.loss_params['contrastive_loss_weight'])

                elif len(labels_dict) > 0:
                    for lf_idx, lf_name in enumerate(labels_dict):
                     
                        lf_labels = labels_dict[lf_name]

                        self.log.losses["contrastive_" + lf_name] = compute_contrastive_loss(posterior.mean, posterior.mean, 
                            lf_idx, labels = lf_labels, temperature = self.loss_params['contrastive_temperature'],
                            base_temperature = self.loss_params['contrastive_base_temperature'],
                            loss_weight = self.loss_params['contrastive_loss_weight'])

        if embed:
            return self.log, enc_idx,z_q,pos #, p_to_a, n_to_a
        return self.log

    def generate_rollout(self, states, horizon):
        rollout_states = [states[0].unsqueeze(0)]
        rollout_actions = []

        T = horizon

        for t in range(T):
            with torch.no_grad():
                curr_state = rollout_states[-1].squeeze(0)
                action = self.act(curr_state)
                next_state = curr_state + action

            rollout_states.append(next_state.unsqueeze(0))
            rollout_actions.append(action.unsqueeze(0))

        rollout_states = torch.cat(rollout_states, dim=0)
        rollout_actions = torch.cat(rollout_actions, dim=0)

        return rollout_states, rollout_actions


    def decode_action(self, state):
        dec_fc_input = torch.cat([state, self.z], dim=1).to(torch.float32)

        if self.is_recurrent:
            dec_fc_input = torch.cat([dec_fc_input, self.hidden[-1]], dim=1)

        dec_h = self.dec_action_fc(dec_fc_input)
        dec_mean = self.dec_action_mean(dec_h)

        if isinstance(self.dec_action_logvar, nn.Parameter):
            dec_logvar = self.dec_action_logvar
        else:
            dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)
