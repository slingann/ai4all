from dgllife.data import BACE
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import SMILESToBigraph
from dgllife.model import load_pretrained

from rdkit import Chem

import torch
from torch.nn import functional as F

node_featurizer = CanonicalAtomFeaturizer()
s2g = SMILESToBigraph(add_self_loop=True, node_featurizer=node_featurizer, edge_featurizer=None)
bace_model = load_pretrained("GCN_canonical_BACE")
bace_model.eval()

def predict_bace(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return bace_model(s2g(smiles), feats=node_featurizer(mol)["h"]).item()


from molgpt.generate.model import GPT, GPTConfig

vocab_size = 94
block_size = 100
n_layer = 8
n_head = 8
n_embed = 256

device = "cuda"

mconf = GPTConfig(vocab_size, block_size, num_props=0,
                  n_layer=n_layer, n_head=n_head, n_embd=n_embed,
                  scaffold=False, scaffold_maxlen=1,
                  lstm=False, lstm_layers=2)
model = GPT(mconf).to(device)
state_dict = torch.load("guacamol_nocond.pt")
del state_dict["prop_nn.weight"]
del state_dict["prop_nn.bias"]
model.load_state_dict(state_dict)
vocab = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

def gpt_generate(n_samples):
    batch_size = n_samples
    x = torch.tensor([20], dtype=torch.long)[None,...].repeat(batch_size, 1).to(device)
    y = sample(model, x, block_size, temperature=1, sample=True, top_k=None, prop = None, scaffold = None)
    smiles = ["".join([vocab[char] for char in sampled]).replace("<", "")
              for sampled in y]
    return smiles

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, prop = None, scaffold = None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _, _ = model(x_cond, prop = prop, scaffold = scaffold)   # for liggpt
        # logits, _, _ = model(x_cond)   # for char_rnn
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x
