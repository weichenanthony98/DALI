from utils.gpu import allocate_cuda_idx, choose_device, get_time
from typing import Any, Union
from collections import Counter
import numpy as np
import torch
from utils.network import get_network


class NeuralEmbeddingSpecification():
    """Learnware Specification via Dual Alignment"""
    def __init__(self, choose_method: float = None, cuda_idx: int = None):
        self.z = None
        self.label = None
        self.model = 'ConvNet_1D'
        self.choose_method = choose_method
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        torch.cuda.empty_cache()
        self._device = choose_device(cuda_idx=self._cuda_idx)


    def assign_indices_class(self, labels, n_classes):
        indices_class = [[] for c in range(n_classes)]
        for i, lab in enumerate(labels):
            indices_class[lab].append(i)
        return indices_class


    def get_data_shuffle(self, dataset, indices_class, c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return dataset[idx_shuffle]


    def generate_state_spec_from_data(
        self,
        X: np.ndarray,
        K: int,
        channel: int,
        true_labels: np.ndarray,
        pseudo_labels: np.ndarray,
        Iteration: int = 100,
        BatchSize: int = 32,
    ):
        num_classes = len(np.unique(true_labels))
        X = torch.tensor(X, dtype=torch.float32).to(self._device)
        true_labels = torch.tensor(true_labels, dtype=torch.long).to(self._device)
        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self._device)

        true_indices_class = self.assign_indices_class(true_labels, num_classes)
        pseudo_indices_class = self.assign_indices_class(pseudo_labels, num_classes)
        init_z = torch.rand(size=(num_classes * K, channel), dtype=torch.float32, requires_grad=True, device=self._device)
        self.label = torch.tensor([np.ones(K) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=self._device).view(-1)
        for c in range(num_classes):
            init_z.data[c * K:(c + 1) * K] = self.get_data_shuffle(X, true_indices_class, c, K).detach().data

        # self.initial_dataspc = self.z.data
        optimizer = torch.optim.Adam([init_z, ], lr=1e-2)
        label_criterion = torch.nn.CrossEntropyLoss().to(self._device)
        best_loss = 1000

        for iter in range(Iteration + 1):
            distribution_net = get_network(self.model, channel, num_classes).to(self._device)
            distribution_net.eval()
            for param in list(distribution_net.parameters()):
                param.requires_grad = False
            embed = distribution_net.module.embed if torch.cuda.device_count() > 1 else distribution_net.embed

            discriminative_net = get_network(self.model, channel=channel, num_classes=num_classes).to(self._device)
            discriminative_net.eval()
            for param in list(discriminative_net.parameters()):
                param.requires_grad = False
            clf = discriminative_net.module.clf if torch.cuda.device_count() > 1 else discriminative_net.clf

            loss_avg = 0
            loss = torch.tensor(0.0).to(self._device)
            real_data_all = []
            real_label_all = []
            pseudo_data_all = []
            pseudo_labels_all = []
            spec_data_all = []
            spec_label_all = []


            for c in range(num_classes):
                real_data = self.get_data_shuffle(X, true_indices_class, c, BatchSize)
                pseudo_data = self.get_data_shuffle(X, pseudo_indices_class, c, BatchSize)
                real_data = torch.tensor(real_data).to(self._device)
                pseudo_data = torch.tensor(pseudo_data).to(self._device)
                spc_data = init_z[c * K:(c+1) * K]

                real_data_all.append(real_data)
                real_label_all.append(torch.ones(real_data.shape[0], dtype=torch.long, requires_grad=False, device=self._device))
                pseudo_data_all.append(pseudo_data)
                pseudo_labels_all.append(torch.ones(pseudo_data.shape[0], dtype=torch.long, requires_grad=False, device=self._device))
                spec_data_all.append(spc_data)
                spec_label_all.append(self.label[c * K:(c+1) * K])

            real_data_all = torch.cat(real_data_all, dim=0).to(self._device)
            pseudo_data_all = torch.cat(pseudo_data_all, dim=0).to(self._device)
            spec_data_all = torch.cat(spec_data_all, dim=0).to(self._device)
            real_label_all = torch.cat(real_label_all, dim=0).to(self._device)
            pseudo_labels_all = torch.cat(pseudo_labels_all, dim=0).to(self._device)
            spec_label_all = torch.cat(spec_label_all, dim=0).to(self._device)

            real_data_all = real_data_all.unsqueeze(0).permute(1, 2, 0)
            spec_data_all = spec_data_all.unsqueeze(0).permute(1, 2, 0)
            pseudo_data_all = pseudo_data_all.unsqueeze(0).permute(1, 2, 0)

            embed_real = embed(real_data_all).detach()
            embed_spec = embed(spec_data_all)
            logits_real = clf(pseudo_data_all).detach()
            logits_spec = clf(spec_data_all)

            discriminative_loss = torch.sqrt((label_criterion(logits_real, pseudo_labels_all) - label_criterion(logits_spec, spec_label_all)) ** 2 + 1e-6)
            distribution_loss = torch.sum((torch.mean(embed_real.reshape(num_classes, int(embed_real.shape[0] / num_classes), -1), dim=1) - torch.mean(embed_spec.reshape(num_classes, K, -1), dim=1)) ** 2)
            loss = discriminative_loss + distribution_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            loss_avg /= (num_classes)
            if iter % 100 == 0:
                print('%s iter = %05d, loss = %.4f, inside_loss = %.4f, DM_loss = %.4f' % (get_time(), iter, loss_avg,discriminative_loss.item(), distribution_loss.item() ))

            if loss_avg < best_loss:
                best_loss = loss_avg
                self.z = init_z





