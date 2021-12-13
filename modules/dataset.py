#-*-coding:UTF-8-*-
import logging
import random
import torch
import re
from cached_property import cached_property
from torch.utils.data import Dataset
from modules.IO import read_raw_amr_data
from modules.linearization import AMRTokens, AMRLinearizer
from pathlib import Path
from modules import ROOT

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
    ):
        self.paths = paths
        self.INIT=tokenizer.INIT
        self.encoder=tokenizer.encoder
        self._tok_bpe=tokenizer._tok_bpe
        self.tokenizer = tokenizer
        self.device = device
        self.max_src_length=100
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.edgevocab=[]
    
    
        for g in graphs:
            l, e = self.tokenizer.linearize(g) 
            
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')
            
            
            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

        for tok in Path(ROOT/'data/vocab/edges.txt').read_text().strip().splitlines():
            self.edgevocab.append(self.INIT+tok)


    def __len__(self): 
        return len(self.sentences) 
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample): 
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x_b = [s['sentences'] for s in samples]
        x_b, extra = self.tokenizer.batch_encode_sentences(x_b, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y_b = [s['linearized_graphs_ids'] for s in samples]
            y_b, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y_b, samples, device=device)
            extra.update(extra_y)
        else:
            y_b = None
        extra['ids'] = [s['id'] for s in samples]

                
       
        graphs=extra['graphs']
        pointers=extra['pointers']
        edges_list=extra['edges_id']
        pointers_var_nodes=extra['pointers_var_nodes']
        extra['enc_edge_lens']=[]
        extra['enc_edge_links1']=[]
        extra['enc_edge_links2']=[]
        extra['recon_edge_neglinks2']=[]
        extra['recon_edge_neglinks1']=[]
        extra['enc_edges_ids']=[]
        extra['recon_edge_dists']=[]
        extra['recon_edge_distids1']=[]
        extra['recon_edge_distids2']=[]
        extra['enc_seq_lens']=[]
        #print(extra)
        


        
        for ii ,g in enumerate(graphs):
            g.metadata={}
            tmp=[[-1] * self.max_src_length*5 for _ in range(self.max_src_length*5)]
            tmp2 = [[0] * self.max_src_length*5 for _ in range(self.max_src_length*5)]
            
            

            lx=len(pointers[ii])

            enc_edge_id = []
            enc_edge_link1 = []
            enc_edge_link2 = []
            enc_seq_mask = [[0] * self.max_src_length for _ in range(self.max_src_length)]
            enc_seq_mask_rev = [[0] * self.max_src_length for _ in range(self.max_src_length)]

            recon_edge_distid1 = []
            recon_edge_distid2 = []
            recon_edge_dist = []
            recon_edge_neglink2 = []
            recon_edge_neglink1 = []
            edge_vacab={}
            
            for k in edges_list[ii].keys():
                if k in self.edgevocab:
                    for v in edges_list[ii][k]:
                        edge_vacab[v]=self.edgevocab.index(k)
                else:
                    continue
                   
       
            dit={}
            g1=[]
            num=1
            for (source,role,target)in g.instances():

                if target in dit.values():
                    num+=1
                    dit[source]=target+str(num)
                else:
                    dit[source]=target
                    
            #print(g.triples)
        
            for i,(n1 , e, n2) in enumerate(g.triples):
                is_not_instance= e!=':instance'
                n1_in_dit= n1 in dit
                n2_in_dit= n2 in dit
                # is_not_wiki= e!=':wiki'
                #is_not_vaule= e!=':value'
                #is_not_quant= e!=':quant'
                value = re.compile(r'[A-Za-z]')
                is_pointer=value.match(n2)
                is_not_mode=e!=':mode'
                special=n2.startswith('"') and n2.endswith('"')
                if is_pointer and is_not_mode:
                    if special:
                        continue
                    if is_not_instance and n1_in_dit and n2_in_dit:
                        n1=dit[n1]
                        n2=dit[n2]
                        tt=(self.INIT+n1,self.INIT+e,self.INIT+n2)
                        g1.append(tt)
                    elif  is_not_instance and n1_in_dit and not n2_in_dit:
                        n1=dit[n1]
                        tt=(self.INIT+n1,self.INIT+e,self.INIT+n2)
                        g1.append(tt)
                    elif is_not_instance and n2_in_dit and not n1_in_dit:
                        n2=dit[n2]
                        tt=(self.INIT+n1,self.INIT+e,self.INIT+n2)
                        g1.append(tt)
                    else:
                        continue
                else:
                   continue
            #print("g1===========")
            #print(g1)
            # node=[]
            # edges=[]
            # for i, (node1, rel, node2) in enumerate(g1):
            #     if  node1 not in node :
            #         node.append(node1)
            #     if node2 not in node :
            #         node.append(node2)
            #     if rel not in edges:
            #         edges.append(rel)
            #print((node1, rel, node2))
            #print('优化后图形的最终的长度: '+str(len(g1)))
            #print(node)
            #print('node的长度为�?'+str(len(node)))
            #print(edges)
            #print('edge的长度为�?'+str(len(edges)))
        
            g2=[]
            #print("tokens===========")
            #print(extra['linearized_graphs'][ii])
            
            #print(pointers_var_nodes[ii])
            #print("g1===========")
            #print(g1)
            #g2_node1_bpe=[]
            #g2_node2_bpe=[]
            for (node1, rel, node2) in g1:
                node1_is_frame = re.match(r'.+-\d\d', node1) is not None
                node2_is_frame = re.match(r'.+-\d\d', node2) is not None
                if node1 in pointers_var_nodes[ii].keys():
                    node1=pointers_var_nodes[ii][node1]
                elif node1_is_frame:
                    node1_bpe=self._tok_bpe(node1.strip(self.INIT), add_space=True)
                    #g2_node1_bpe.append(node1_bpe)
                    if node1_bpe[0] in pointers_var_nodes[ii].keys():
                        node1=pointers_var_nodes[ii][node1_bpe[0]]
                    else:
                        continue

                else:
                    node1_bpe=self._tok_bpe(node1.strip(self.INIT),add_space=True)
                    #g2_node1_bpe.append(node1_bpe)
                    if node1_bpe[0] in pointers_var_nodes[ii].keys():
                        node1=pointers_var_nodes[ii][node1_bpe[0]]
                    else:
                        continue
               
                if node2 in  pointers_var_nodes[ii].keys():
                    node2=pointers_var_nodes[ii][node2]
                elif node2_is_frame:
                    node2_bpe=self._tok_bpe(node2.strip(self.INIT), add_space=True) 
                    #g2_node2_bpe.append(node2_bpe)
                    if node2_bpe[0] in pointers_var_nodes[ii].keys():
                        node2=pointers_var_nodes[ii][node2_bpe[0]]
                    else:
                        continue

                else:
                    node2_bpe=self._tok_bpe(node2.strip(self.INIT),add_space=True)
                    #g2_node2_bpe.append(node2_bpe)
                    if node2_bpe[0] in pointers_var_nodes[ii].keys():
                        node2=pointers_var_nodes[ii][node2_bpe[0]]
                    else:
                        continue
                g2.append((node1, rel, node2))
                
            #print('替换pointer后的g2�?\n')
            #print(g2_node1_bpe)
            #print(g2_node2_bpe)
            #print("g2===========")
            #print(g2)
            # for (p1,e,p2) in g2:
            #     print((p1,e,p2))
            g3=[]    
   
            for (p1,e,p2) in g2: 
                m=0
                for i in pointers[ii][p1]:
                    tmpa=int(i)
                    for j in pointers[ii][p2]:
                        tmpb=int(j)
                        mi=min(tmpa,tmpb)
                        ma=max(tmpa,tmpb)

                        if e in edges_list[ii]:

                            for k in edges_list[ii][e]:
                                tmpc=int(k)

                                if tmpc>=mi and tmpc<=ma and (tmpc-mi)>=(ma-tmpc):
                                    (new1, newe, new2)= tmpa, tmpc,tmpb
                                    m+=1
                                    g3.append((new1, newe, new2))
                                    if m>=1:
                                        break
                            if m>=1:
                                break
                        else:
                            break

               
                    if m>0:
                        break
                    
            g4=[]
            for (p1,e,p2) in g3:
                if e in edge_vacab:
                    e=edge_vacab[e]
                    g4.append((p1,e,p2))
                else:
                    continue
    

            j=0
            
            for k, (node1, rel, node2) in enumerate(g4):



                id1=node1
                id2=node2
                edge=rel

                if id1 >= self.max_src_length - 1 or id2 >= self.max_src_length - 1:
                    continue
                enc_edge_id.append(edge) 
                enc_edge_link1.append(id1)
                enc_edge_link2.append(id2)

                x=random.randint(0,lx-1)
                if x == id2:
                    x = (x+1) % lx
                recon_edge_neglink2.append(x)
                x = random.randint(0, lx - 1)
                if x == id1:
                    x = (x + 1) % lx
                recon_edge_neglink1.append(x)
                tmp[id1][id2] = 1
                tmp[id2][id1] = 1
                tmp2[id1][id2] = 1
                tmp2[id2][id1] = -1
                j += 1
                if j >= self.max_src_length:
                    break
                    

            le = len(enc_edge_id)
            
            t = 0
            for k in range(lx):
                tmp[k][k] = 0
                for i in range(lx):                    
                    if tmp[i][k] == -1:
                        continue
                    for j in range(lx):
                        if tmp[k][j] == -1:
                            continue
                        if i == j:
                            continue
                       
                        if tmp[i][j] == -1 or tmp[i][k] + tmp[k][j] < tmp[i][j]:
                            tmp[i][j] = tmp[i][k] + tmp[k][j]

                        t += 1
                        if t >= self.max_src_length * 1:
                            break
                    if t >= self.max_src_length * 1:
                        break
                if t >= self.max_src_length * 1:
                    break




            for i in range(lx):
                x = random.randint(0, lx - 1)
                y = random.randint(0, lx - 1)
                recon_edge_distid1.append(x)
                recon_edge_distid2.append(y)
                if tmp[x][y] == -1 or tmp[x][y] >=13:
                    recon_edge_dist.append(13)
                else:
                    recon_edge_dist.append(tmp[x][y])
                




            extra['enc_edge_lens'].append(le)
            extra['enc_seq_lens'].append(lx)
            extra['enc_edge_links1'].append(enc_edge_link1+[0]*(self.max_src_length * 1 - le))
            extra['enc_edge_links2'].append(enc_edge_link2+[0]*(self.max_src_length * 1 - le))
            extra['recon_edge_neglinks2'].append(recon_edge_neglink2+[0]*(self.max_src_length * 1 - le))
            extra['recon_edge_neglinks1'].append(recon_edge_neglink1+[0]*(self.max_src_length * 1 - le))
            extra['enc_edges_ids'].append(enc_edge_id+[0]*(self.max_src_length * 1 - le))
            extra['recon_edge_dists'].append(recon_edge_dist + [0] * (self.max_src_length * 1 - lx))
            extra['recon_edge_distids1'].append(recon_edge_distid1 + [0] * (self.max_src_length * 1 - lx))
            extra['recon_edge_distids2'].append(recon_edge_distid2 + [0] * (self.max_src_length * 1 - lx))
            #print("+======================================+")
            #print(extra)
            #print(extra['enc_edge_links1'])


        return x_b, y_b, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler() 
        it = ([[self.dataset[s] for s in b] for b in it]) 
        it = (self.dataset.collate_fn(b, device=self.device) for b in it) 
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
           
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx]) 
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps 
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
