import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import math
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")     


class NPregression(nn.Module):
    def __init__(self, kind, dim_input, dim_output, dim_hidden=128, num_heads=8):
        super(NPRegression, self).__init__()
        self.kind = kind
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        if 'CNP' in self.kind:
            dec_dim = dim_input + dim_hidden
        else:
            dec_dim = dim_input + dim_hidden * 2
            
        self.x_MLP = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden))
        
        if 'LD' in self.kind:
            r_dim = dim_hidden * dim_output * 2
            z_dim = dim_hidden * dim_output * 2
            
            self.param_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                           nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden))
        
            self.layernorm = nn.LayerNorm(dim_hidden)
            
        else:
            r_dim = dim_hidden
            z_dim = dim_hidden
            self.decoding = nn.Sequential(nn.Linear(dec_dim, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_output * 2))
    
        
        if 'ANP' in self.kind:
            self.e_MLP = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, r_dim))
        
            self.cross_Attention = nn.MultiheadAttention(r_dim, num_heads)
        
            self.r_Attention = nn.MultiheadAttention(r_dim, num_heads)
            
            self.sub_Layer = nn.Sequential(nn.Linear(r_dim, r_dim * 2),
                                           nn.ReLU(),
                                           nn.Linear(r_dim * 2, r_dim))
            
            self.layer_Norm = nn.LayerNorm(r_dim)
        
            self.s_Attention = nn.MultiheadAttention(dim_hidden, num_heads)
        
        self.r_MLP = nn.Sequential(nn.Linear(dim_hidden + dim_output, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, r_dim))
        
        self.s_MLP = nn.Sequential(nn.Linear(dim_hidden + dim_output, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden))
            
        self.z_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, z_dim * 2))
        
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def deterministic_encoder(self, det_enc_inp, Cx=None, Tx=None, attention=False, synthetic=True):
        if attention == False:
            ri = self.r_MLP(det_enc_inp)
            r = ri.mean(1, True)
        else:
            if synthetic:
                _, num_context, _ = det_enc_inp.size()
                query = self.e_MLP(Tx).permute(1,0,2)
                key = self.e_MLP(Cx).permute(1,0,2)
                value = self.r_MLP(det_enc_inp).permute(1,0,2)
                out, _ = self.cross_Attention(query, key, value)
                r = out.permute(1,0,2).unsqueeze(2)
            else:
                hidden1 = self.r_MLP(det_enc_inp).permute(1,0,2)
                hidden2, _ = self.r_Attention(hidden1, hidden1, hidden1)
                ri, _ = self.r_Attention(hidden2, hidden2, hidden2)
                
                query = self.e_MLP(Tx).permute(1,0,2)
                key = self.e_MLP(Cx).permute(1,0,2)
                out, _ = self.cross_Attention(query, key, ri)
                sub1 = out + query
                sub2 = self.layer_Norm(sub1.permute(1,0,2))
                r = self.layer_Norm(self.sub_Layer(sub2)+sub2).unsqueeze(2)
        return r        
        
    def reparameterize(self, dist):
        eps = torch.randn(size = dist.stddev.size()).to(device)
        sample = dist.mean + dist.stddev * eps
        return sample
    
    def stochastic_distribution(self, z_param):
        half = int(z_param.size(-1)/2)
        z_mu = z_param[:, :, :half]
        z_omega = z_param[:, :, half:]
        z_sigma = 0.1+0.9*self.sigmoid(z_omega)
        z_dist = torch.distributions.normal.Normal(z_mu, z_sigma)
        return z_dist
    
    def stochastic_encoder(self, lat_enc_inp, attention=False):
        if attention == False:
            si = self.s_MLP(lat_enc_inp)
            s = si.mean(1, True)
            z_param = self.z_MLP(s)
        else:
            hidden1 = self.s_MLP(lat_enc_inp).permute(1,0,2)
            hidden2, _ = self.s_Attention(hidden1, hidden1, hidden1)
            si, _ = self.s_Attention(hidden2, hidden2, hidden2)
            s = si.permute(1,0,2).mean(1, True)
            z_param = self.z_MLP(s)
        z_dist = self.stochastic_distribution(z_param)
        return z_dist
    
    def decoder(self, x, r, z, linear):
        if linear == False:
            r, z = r.squeeze(2), z.squeeze(2)
            dec_inp = torch.cat((x, r, z), 2)
            param = self.decoding(dec_inp)
            mu = param[:, :, :self.dim_output]
            omega = param[:, :, self.dim_output:]
            sigma = 0.1+0.9*self.softplus(omega)
            dist = torch.distributions.normal.Normal(mu, sigma)
        else:
            r = r.reshape(-1, r.size(1), self.dim_output * 2, self.dim_hidden)
            z = z.reshape(-1, z.size(1), self.dim_output * 2, self.dim_hidden)
            dec_param = r + self.param_MLP(z)
            dec_param = self.layernorm(dec_param).permute(0,1,3,2)
            
            param = torch.matmul(self.x_MLP(x).unsqueeze(2), dec_param).squeeze()
            mu = param[:,:,:self.dim_output]
            omega = param[:,:,self.dim_output:]
            sigma = 0.1+0.9*self.softplus(omega)
            dist = torch.distributions.normal.Normal(mu, sigma)
        return dist

    def forward(self, Cx, Cy, Tx, Ty, synthetic=True):
        _, num_context, _ = Cx.size()
        _, num_target, _ = Ty.size()
        
        context = torch.cat((self.x_MLP(Cx), Cy), 2)
        target = torch.cat((self.x_MLP(Tx), Ty), 2)
        
        NLL = torch.tensor([0.0]).to(device)
        
        if 'LD' in self.kind:
            linear = True
        else:
            linear = False
        
        if 'CNP' in self.kind:
            r_c = self.deterministic_encoder(context)
            r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
            r_t = self.deterministic_encoder(target)
            r_t = r_t.unsqueeze(1).repeat(1, num_target, 1, 1)
            
            z = torch.empty(0).to(device)
            
            KL = torch.tensor([0.0]).to(device)
        
        elif 'ANP' in self.kind:
            r_c = self.deterministic_encoder(context, Cx, Tx, attention=True, synthetic=synthetic)
            r_t = self.deterministic_encoder(target, Tx, Tx, attention=True, synthetic=synthetic)

            z_prior = self.stochastic_encoder(context, attention=True)               
            z_posterior = self.stochastic_encoder(target, attention=True)      
            if self.training:
                z_sample = self.reparameterize(z_posterior)
            else:
                z_sample = z_prior.mean
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)
            
            KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
            
        elif 'NP' in self.kind:
            r_c = self.deterministic_encoder(context)
            r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
            r_t = self.deterministic_encoder(target)
            r_t = r_t.unsqueeze(1).repeat(1, num_target, 1, 1)
                
            z_prior = self.stochastic_encoder(context)
            z_posterior = self.stochastic_encoder(target)
            if self.training:
                z_sample = self.reparameterize(z_posterior)
            else:
                z_sample = z_prior.mean
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)

            KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
                    
        y_dist = self.decoder(Tx, r_c, z, linear=linear)          
        NLL += -y_dist.log_prob(Ty).sum(-1).mean()   
        MSE = ((Ty-y_dist.mean)**2).sum(-1).mean()
        
        return y_dist, KL, NLL, MSE

    
class NPclassification(nn.Module):
    def __init__(self, kind, dim_input, dim_output, dim_hidden=128, num_heads=8, dp_rate=0.0):
        super(NPClassification, self).__init__()
        self.kind = kind
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        r_dim = dim_hidden
        z_dim = dim_hidden
        
        if 'MetaFun' in self.kind:
            dec_dim = dim_hidden
        elif 'CNP' in self.kind:
            dec_dim = dim_hidden * 6
        else:
            dec_dim = dim_hidden * 11
        
        if 'LD' in self.kind:
            self.param_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                           nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden))
        
            self.layernorm = nn.LayerNorm(dim_hidden)
            
        else:
            self.decoding = nn.Sequential(nn.Linear(dec_dim, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_output))
    
            
        if dim_input == 84*84*3:
            self.x_emb = ImageEmbedding(dim_hidden, dp_rate)
            
        else:
            self.x_emb = nn.Sequential(nn.Dropout(p=dp_rate),
                                       nn.Linear(dim_input, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
        
        if 'ANP' in self.kind:
            self.cross_Attention = nn.MultiheadAttention(dim_hidden, num_heads)
        
            self.r_Attention = nn.MultiheadAttention(dim_hidden, num_heads)
            
            self.s_Attention = nn.MultiheadAttention(dim_hidden, num_heads)
        
        
        if 'MetaFun' in self.kind:
            self.l1 = nn.Parameter(torch.Tensor([1]*dim_hidden))
        
            self.m_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
            
            self.u_t_MLP = nn.Sequential(nn.Linear(dim_hidden * 2, dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden, dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden, dim_hidden))
            
            self.u_f_MLP = nn.Sequential(nn.Linear(dim_hidden * 2, dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden, dim_hidden),
                                         nn.ReLU(),
                                         nn.Linear(dim_hidden, dim_hidden))
            
            self.w_MLP = nn.Sequential(nn.Linear(dec_dim, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden * 2))
        
        else:
            self.r_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, r_dim))
        
        if 'ANP' in self.kind or 'MetaFun' in self.kind:
            self.e_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
            
        if 'NP' in self.kind or 'ANP' in self.kind:
            self.s_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, dim_hidden))
            
            self.z_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, z_dim * 2))
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def kernel(self, a, b):
        a = torch.matmul(torch.unsqueeze(a, 2), (1/self.l1).diag())
        b = torch.matmul(torch.unsqueeze(b, 1), (1/self.l1).diag())
        norm = torch.norm(a-b, dim=-1)
        norm_squared = torch.pow(norm, 2)
        kernel = torch.exp(-0.5 * norm_squared)
        return kernel
    
    def deterministic_encoder(self, x, y, t0=None, attention=False, iterative=False):
        if iterative:
            tmp = torch.zeros((x.size(0), x.size(1), self.dim_output, self.dim_hidden)).to(device)
            r = torch.zeros((x.size(0), t0.size(1), self.dim_output, self.dim_hidden)).to(device)
                            
            query = self.e_MLP(t0)
            key = self.e_MLP(x)
                
            for iter_idx in range(5): 
                m_k = self.m_MLP(tmp)
                m = m_k.mean(2, True).repeat(1, 1, self.dim_output, 1)
                u_t = self.u_t_MLP(torch.cat((m_k, m), 3))
                u_f = self.u_f_MLP(torch.cat((m_k, m), 3))
                    
                u = y.unsqueeze(3) * u_t + (1-y.unsqueeze(3)) * u_f
                weight_c = self.kernel(key, key)
                weight_t = self.kernel(query, key)
                for class_idx in range(self.dim_output):
                    delta_r_c = torch.matmul(weight_c, u[:, :, class_idx, :])
                    tmp[:, :, class_idx, :] -= 0.1 * delta_r_c                            
                    delta_r_t_ = torch.matmul(weight_t, u[:, :, class_idx, :])
                    r[:, :, class_idx, :] -= 0.1 * delta_r_t
        
        else:
            r = torch.tensor([]).to(device)
            for batch_idx in range(x.size(0)):
                class_param = torch.tensor([]).to(device)
                for class_idx in range(self.dim_output):
                    index = class_idx == y[batch_idx].max(-1).indices
                    inp = x[batch_idx][index].reshape(1, -1, self.dim_hidden)    
                    if attention == False:
                        ri = self.r_MLP(inp).mean(1, True)
                    else:
                        query = self.e_MLP(t0[batch_idx:batch_idx+1]).permute(1,0,2)
                        key = self.e_MLP(inp).permute(1,0,2)
                        value = self.r_MLP(inp).permute(1,0,2)
                        out, _ = self.cross_Attention(query, key, value)
                        ri = out.permute(1,0,2).unsqueeze(2)
                    class_param = torch.cat((class_param, ri), -2)
                r = torch.cat((r, class_param), 0)
        return r        
        
    def reparameterize(self, dist):
        eps = torch.randn(size = dist.stddev.size()).to(device)
        sample = dist.mean + dist.stddev * eps
        return sample
    
    def stochastic_distribution(self, z_param):
        z_mu = z_param[:, :, :self.dim_hidden]
        z_omega = z_param[:, :, self.dim_hidden:]
        z_sigma = 0.1+0.9*self.sigmoid(z_omega)
        z_dist = torch.distributions.normal.Normal(z_mu, z_sigma)
        return z_dist
    
    def stochastic_encoder(self, x, y, attention=False:
        z_param = torch.tensor([]).to(device)
        for batch_idx in range(x.size(0)):
            class_param = torch.tensor([]).to(device)
            for class_idx in range(self.dim_output):
                index = class_idx == y[batch_idx].max(-1).indices
                inp = x[batch_idx][index].reshape(1, -1, self.dim_hidden)
                if attention == False:
                    si = self.s_MLP(inp).mean(1, True)
                    zi_param = self.z_MLP(si)
                else:
                    hidden1 = self.s_MLP(inp).permute(1,0,2)
                    hidden2, _ = self.s_Attention(hidden1, hidden1, hidden1)
                    si, _ = self.s_Attention(hidden2, hidden2, hidden2)
                    si = si.permute(1,0,2).mean(1, True)
                    zi_param = self.z_MLP(si)
                class_param = torch.cat((class_param, zi_param), 1)
            z_param = torch.cat((z_param, class_param), 0)
        return z_param
            
    def decoder(self, x, r, z, iterative=False, linear=True):
        if iterative:
            w_param = self.w_MLP(r)
            w_mu = w_param[:, :, :, :self.dim_hidden]
            w_omega = w_param[:, :, :, self.dim_hidden:]
            w_sigma = 0.1+0.9*self.softplus(w_omega)
        
            w_dist = torch.distributions.normal.Normal(w_mu, w_sigma)
            w = self.reparameterize(w_dist).permute(0,1,3,2)
            logits = torch.matmul(x.unsqueeze(2), w).squeeze()
        
        elif linear:
            if z == None:
                w = r
            else:
                w = r + self.param_MLP(z)
            w = self.layernorm(w).permute(0,1,3,2)
            logits = torch.matmul(x.unsqueeze(2), w).squeeze()
        
        else:
            if z == None:
                dec_inp = torch.cat((x, r), 2)
            else:
                dec_inp = torch.cat((x, r, z), 2)
            logits = self.decoding(dec_inp)
            
        dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)    
        return dist

    def forward(self, Cx, Cy, Tx, Ty):
        c0 = self.x_emb(Cx)
        t0 = self.x_emb(Tx)
        
        _, num_context, _ = c0.size()
        _, num_target, _ = t0.size()
        
        NLL = torch.tensor([0.0]).to(device)
        
        iterative=False
        linear = False
        
        if 'CNP' in self.kind:
            r_c = self.deterministic_encoder(c0, Cy)
            r_t = self.deterministic_encoder(t0, Ty)
            
            if 'LD' in self.kind:
                linear = True
                r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
                r_t = r_t.unsqueeze(1).repeat(1, num_target, 1, 1)
                
            else:
                r_c = r_c.reshape(r_c.size(0), 1, -1)
                r_c = r_c.repeat(1, num_target, 1)
                r_t = r_t.reshape(r_t.size(0), 1, -1)
                r_t = r_t.repeat(1, num_target, 1)
            
            KL = torch.tensor([0.0]).to(device)
            
            z = None
            
        elif 'ANP' in self.kind:
            r_c = self.deterministic_encoder(c0, Cy, t0, attention=True)
            r_t = self.deterministic_encoder(t0, Ty, t0, attention=True)
            
            z_c_param = self.stochastic_encoder(c0, Cy, attention=True)             
            z_prior = self.stochastic_distribution(z_c_param)
            z_t_param = self.stochastic_encoder(t0, Ty, attention=True)      
            z_posterior = self.stochastic_distribution(z_t_param)
            
            if self.training:
                z_sample = self.reparameterize(z_posterior)
            else:
                z_sample = z_prior.mean
            
            if 'LD' in self.kind:
                linear = True
                z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)
            else:
                r_c = r_c.reshape(r_c.size(0), num_target, -1)
                r_t = r_t.reshape(r_t.size(0), num_target, -1)
                
                z_sample = z_sample.reshape(z_sample.size(0), 1, -1)
                z = z_sample.repeat(1, num_target, 1)
            
            KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
                        
        elif 'NP' in self.kind:
            r_c = self.deterministic_encoder(c0, Cy)
            r_t = self.deterministic_encoder(t0, Ty)
            
            z_c_param = self.stochastic_encoder(c0, Cy)             
            z_prior = self.stochastic_distribution(z_c_param)
            z_t_param = self.stochastic_encoder(t0, Ty)      
            z_posterior = self.stochastic_distribution(z_t_param)
            
            if self.training:
                z_sample = self.reparameterize(z_posterior)
            else:
                z_sample = z_prior.mean
            
            if 'LD' in self.kind:
                linear = True
                r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
                r_t = r_t.unsqueeze(1).repeat(1, num_target, 1, 1)
                z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)
            else:
                r_c = r_c.reshape(r_c.size(0), 1, -1)
                r_c = r_c.repeat(1, num_target, 1)
                r_t = r_t.reshape(r_t.size(0), 1, -1)
                r_t = r_t.repeat(1, num_target, 1)
                
                z_sample = z_sample.reshape(z_sample.size(0), 1, -1)
                z = z_sample.repeat(1, num_target, 1)
                    
            KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
                                
        elif 'MetaFun' in self.kind:
            iterative=True
            r_c = self.deterministic_encoder(c0, Cy, t0, iterative=iterative)
            
            KL = torch.tensor([0.0]).to(device)
            
            z = None
            
        y_dist = self.decoder(t0, r_c, z, iterative, linear)          
        NLL += -y_dist.log_prob(Ty).sum(-1).mean()
        accur = ((Ty.max(-1).indices == y_dist.logits.max(-1).indices).sum(-1) * 100.0 / Ty.size(1)).mean()
        
        return y_dist, KL, NLL, accur
        

    
    
