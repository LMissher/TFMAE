import torch
import numpy as np
import os
from tqdm import tqdm
from model.MTFAE import MTFA
from data_factory.data_loader import get_loader_segment


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.model = MTFA(win_size=self.win_size, seq_size=self.seq_size, c_in=self.input_c, c_out=self.output_c, d_model=self.d_model, e_layers=self.e_layers, fr=self.fr, tr=self.tr, dev=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()

        loss_list = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)

                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                        my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                            tematt[u])))
                    con_loss += (torch.mean(
                        my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                                tematt[u].detach())) + torch.mean(
                        my_kl_loss(tematt[u].detach(),
                                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                adv_loss = adv_loss / len(freatt)
                con_loss = con_loss / len(freatt)

                loss_list.append((con_loss - adv_loss).item())

        return np.average(loss_list)

    def train(self):

        print("======================TRAIN MODE======================")

        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)

        for epoch in tqdm(range(self.num_epochs)):
            loss_list = []

            self.model.train()
            with tqdm(total=train_steps) as pbar:
                for i, (input_data, labels) in enumerate(self.train_loader):

                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)

                    tematt, freatt = self.model(input)

                    adv_loss = 0.0
                    con_loss = 0.0

                    for u in range(len(freatt)):
                        adv_loss += (torch.mean(my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                            my_kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                                    tematt[u])))
                        con_loss += (torch.mean(my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach())) + torch.mean(
                            my_kl_loss(tematt[u].detach(), (
                                    freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                    adv_loss = adv_loss / len(freatt)
                    con_loss = con_loss / len(freatt)

                    loss =  con_loss - adv_loss
                    loss_list.append(loss.item())

                    pbar.update(1)

                    loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)

            vali_loss = self.vali(self.vali_loader)

            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        # (1) find the threshold
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss = my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                    else:
                        adv_loss += my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss += my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature

                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (2) evaluation on the test set
        test_labels = []
        attens_energy = []
        with torch.no_grad():
            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)

                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss = my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                    else:
                        adv_loss += my_kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * temperature
                        con_loss += my_kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * temperature
                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score
