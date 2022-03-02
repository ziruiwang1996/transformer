from transformer import *
from bert_pre_training import Bert

class Generative_model(nn.Module):
    def __init__(self, d_model=20, num_layer=12, heads=5, device='cpu', ff_expansion=4, dropout=0.1, trg_vocab_size=20):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.pre_trained = Bert()
        self.decoder = Decoder(d_model, num_layer, heads, device, ff_expansion, dropout, trg_vocab_size)

    def make_trg_mask(self, trg_seq):
        n, trg_len = len(trg_seq), len(trg_seq[0])
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(n, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, dec_input, segment_pos, src_mask):
        self.pre_trained.load_state_dict(torch.load('pre_trained_model.pth'))
        self.pre_trained.eval()
        enc_out = self.pre_trained(dec_input, {0: 20, 20: 38, 38: 58}, None)
        trg_mask = self.make_trg_mask(dec_input)
        out = self.decoder(dec_input, enc_out, segment_pos, src_mask, trg_mask)
        return out

dec_input = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKDPEYAVYEIDGLPNLTYAQRRAFIVALWDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKEADAAFAEIFKLPNLTFLQFLAFIQALSDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKEAESAEKEIVTLPNLTWWQRLAFILALEDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAEEPEYAVYEIDGLPNLTYAQRRAFIVALWDDPSQSSELLSEAKKLNDSQAPK']

if __name__ == "__main__":
    model = Generative_model()
    out = model(dec_input, {0: 20, 20: 38, 38: 58}, src_mask=None)
    print(out.shape)
