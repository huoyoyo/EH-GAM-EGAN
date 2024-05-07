import torch
import torch.nn as nn
import numpy as np
import os

from model.EHGAMEGAN import Generator, Discriminator, LSTM_AD, GDN
from data_factory.data_loader import get_loader_segment
from utils.utils import *
from utils.optimizer import *
from utils.net_struct import *

from tqdm import tqdm

from src.eval_methods import bf_search
from pprint import pprint


class EarlyStopping:
    def __init__(self, model_save_path, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience 
        self.verbose = verbose  
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_save_path = model_save_path

    
    def __call__(self, val_loss, model_G, model_D, Predictor, Predictor_S, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(self, model_G, model_D, Predictor, Predictor_S, epoch)
        else:
            self.best_score = score
            self.save_model(self, model_G, model_D, Predictor, Predictor_S, epoch)
            self.counter = 0

    def save_model(self, val_loss, model_G, model_D, Predictor, Predictor_S, epoch):
        print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.state_dict(),
            'model_D_state_dict': model_D.state_dict(),
            'model_P_state_dict': Predictor.state_dict(),
            'model_PS_state_dict': Predictor_S.state_dict(),
        }, file_path)
        self.val_loss_min = val_loss





# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        _, self.train_loader = get_loader_segment(self.data_path,
                                                  batch_size=self.batch_size,
                                                  win_size=self.win_size,
                                                  mode='train',
                                                  dataset=self.dataset)
        self.cur_dataset, self.test_loader = get_loader_segment(self.data_path,
                                                                batch_size=self.batch_size,
                                                                win_size=self.win_size,
                                                                mode='test',
                                                                dataset=self.dataset)

        self.device = torch.device(device)
        self.criterion = nn.MSELoss(reduction='none')

        fc_edge_index = generate_fc_edge_index(self.input_c)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = [fc_edge_index]
        self.edge_index_sets = edge_index_sets

    def train(self):
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=self.input_c)
        discriminator = Discriminator(win_size=self.win_size, input_c=self.input_c)
        predictor_T = LSTM_AD(feats=self.input_c)
        predictor_S = GDN(edge_index_sets=self.edge_index_sets,
                          node_num=self.input_c,
                          win_size=self.win_size,
                          out_layer_num=1,
                          out_layer_inter_dim=self.gat_inter_dim
                          )

        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_PT = torch.optim.Adam(predictor_T.parameters(), lr=self.lr)
        optimizer_PS = torch.optim.Adam(predictor_S.parameters(), lr=self.lr, weight_decay=self.decay)

        if torch.cuda.is_available():
            generator.to(device)
            discriminator.to(device)
            predictor_T.to(device)
            predictor_S.to(device)
            generator.train()
            discriminator.train()
            predictor_T.train()
            predictor_S.train()

        print("======================TRAIN MODE======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(path, patience=15, verbose=False, dataset_name=self.dataset)
        rec_losses = []
        p_losses = []
        ps_losses = []
        last_mse = 0
        for epoch in tqdm(range(self.num_epochs)):
            for i, (input_data, y_input_data) in enumerate(self.train_loader):

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                optimizer_PT.zero_grad()
                optimizer_PS.zero_grad()

                input_data = input_data.float().to(self.device)
                y_input_data = y_input_data.float().to(self.device)  # (b,1,n)

                z = torch.FloatTensor(
                    np.random.normal(0, 1, (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))).to(
                    device)
                z = z + y_input_data

                # Generate a batch of input
                fake_input = generator(z)

                p = predictor_T(input_data)
                p_loss = torch.mean(self.criterion(p, y_input_data))
                p_loss.backward()
                optimizer_PT.step()

                ps = predictor_S(input_data, self.edge_index_sets)
                ps_loss = torch.mean(self.criterion(ps, y_input_data))
                ps_loss.backward()
                optimizer_PS.step()

                real_input = y_input_data

                # Real input
                real_validity = discriminator(real_input)
                # Fake input 
                fake_validity = discriminator(fake_input)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input, fake_input)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                # print('2',d_loss.shape)
                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % 1 == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    fake_input = generator(z)

                    fake_validity = discriminator(fake_input)

                    g_loss = -torch.mean(
                        fake_validity)  # + pre_loss # + torch.mean(self.criterion(fake_input, real_input))

                    g_loss.backward()
                    optimizer_G.step()

                    rec_loss = torch.mean(self.criterion(fake_input, real_input))
                    rec_losses.append(rec_loss.detach().cpu().numpy())

                p_losses.append(p_loss.detach().cpu().numpy())
                ps_losses.append(ps_loss.detach().cpu().numpy())
            if epoch % 1 == 0:
                mse = np.average(rec_losses)
                tqdm.write(
                    "Epoch: {0}, Steps: {1} | g_loss Loss: {2:.7f} d_loss Loss: {3:.7f} MSE: {4:.7f} SPD: {5:.7f} P_MSE: {6:.7f} PS_MSE: {6:.7f}".format(
                        epoch + 1,
                        i,
                        g_loss.item(),
                        d_loss.item(),
                        mse,
                        last_mse - mse,
                        np.average(p_losses),
                        np.average(ps_losses)))

                last_mse = mse
            early_stopping(mse, generator, discriminator, predictor_T, predictor_S, epoch)
            if early_stopping.early_stop:
                print("Early stopping with patience ", early_stopping.patience)
                break

    def test(self):
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=self.input_c)
        discriminator = Discriminator(win_size=self.win_size, input_c=self.input_c)
        predictor = LSTM_AD(feats=self.input_c)
        predictor_S = GDN(edge_index_sets=self.edge_index_sets,
                          node_num=self.input_c,
                          win_size=self.win_size,
                          out_layer_num=1,
                          out_layer_inter_dim=self.gat_inter_dim
                          )

        fname = f'{self.model_save_path}_{self.dataset}/model.ckpt'
        map_location = torch.device('cpu')
        checkpoint = torch.load(fname, map_location)

        generator.load_state_dict(checkpoint['model_G_state_dict'])
        discriminator.load_state_dict(checkpoint['model_D_state_dict'])
        predictor.load_state_dict(checkpoint['model_P_state_dict'])
        predictor_S.load_state_dict(checkpoint['model_PS_state_dict'])

        generator.to(device)
        discriminator.to(device)
        predictor.to(device)
        predictor_S.to(device)

        generator.eval()
        discriminator.eval()
        predictor.eval()
        predictor_S.eval()

        print("======================TEST MODE======================")
        criterion = nn.MSELoss(reduction='none')

        test_labels = []
        test_energy = []

        p_energy = []
        ps_energy = []
        g_energy = []
        d_energy = []
        if 'exp' in self.dataset: print('in exp.....')
        for i, (input_data, y_input_data) in enumerate(self.test_loader):
            input_data = input_data.float().to(self.device)  # (b,win_size,n)
            y_input_data = y_input_data.float().to(self.device)  # (b,1,n)

            z = torch.FloatTensor(
                np.random.normal(0, 1, (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))
            ).to(device)  # (b,1,n)
            # Latent Space Z
            z = z + y_input_data

            fake_input = generator(z)
            g_loss = criterion(y_input_data, fake_input)  # (b,w,n)

            p = predictor(input_data)
            p_loss = criterion(p, y_input_data)

            ps = predictor_S(input_data, self.edge_index_sets)
            ps_loss = criterion(ps, y_input_data)

            d_loss = ((discriminator(y_input_data)) * (-1) + 1)

            # loss = self.alpha * g_loss + (1 - self.alpha) * d_loss + self.beta * p_loss + (1 - self.beta) * ps_loss
            loss = self.alpha * ps_loss + (1 - self.alpha) * p_loss + self.beta * g_loss + (1 - self.beta) * d_loss
            # loss = 0.05 * ps_loss + 0.05 * p_loss + 1.33 * g_loss + 0.57 * d_loss
            # loss = p_loss
            loss = torch.mean(loss, dim=1)  # (b,n)
            p_loss = torch.mean(p_loss, dim=1)
            p_loss = torch.mean(p_loss, dim=-1)  # (b)
            ps_loss = torch.mean(ps_loss, dim=1)
            ps_loss = torch.mean(ps_loss, dim=-1)  # (b)
            d_loss = torch.mean(torch.mean(d_loss, dim=1), dim=-1)  # (b)
            g_loss = torch.mean(torch.mean(g_loss, dim=1), dim=-1)  # (b)

            win_loss = torch.mean(loss, dim=-1)  # (b)

            test_energy.append(win_loss.detach().cpu().numpy())
            p_energy.append(p_loss.detach().cpu().numpy())
            ps_energy.append(ps_loss.detach().cpu().numpy())
            d_energy.append(d_loss.detach().cpu().numpy())
            g_energy.append(g_loss.detach().cpu().numpy())

        print('bb', np.concatenate(test_energy, axis=0).shape)

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        test_energy = np.array(test_energy)

        p_energy = np.concatenate(p_energy, axis=0).reshape(-1)
        p_energy = np.array(p_energy)
        ps_energy = np.concatenate(ps_energy, axis=0).reshape(-1)
        ps_energy = np.array(ps_energy)
        g_energy = np.concatenate(g_energy, axis=0).reshape(-1)
        g_energy = np.array(g_energy)
        d_energy = np.concatenate(d_energy, axis=0).reshape(-1)
        d_energy = np.array(d_energy)

        test_labels = self.cur_dataset.test_labels[self.win_size:]

        print("test_labels:     ", test_labels.shape)
        np.save('./energy/test_energy_' + self.dataset, test_energy)
        np.save('./labels/test_labels_' + self.dataset, test_labels)
        np.save('./energy/p_energy_' + self.dataset, p_energy)
        np.save('./energy/p_energy_' + self.dataset, ps_energy)
        np.save('./energy/g_energy_' + self.dataset, g_energy)
        np.save('./energy/d_energy_' + self.dataset, d_energy)

        score = test_energy
        label = test_labels

        print('aa', score.shape, label.shape)
        bf_eval = bf_search(score, label, start=0.001, end=1, step_num=150, verbose=False)
        pprint(bf_eval)
