from transformer import *

# sketch

class Generative_model(nn.Module):
    def __init__(self, d_model=20, num_layer=12, heads=5, device='cpu', ff_expansion=4, dropout=0.1, trg_vocab_size=20):
        super().__init__()
        self.d_model = d_model
        self.decoder = decoder(d_model, num_layer, heads, device, ff_expansion, dropout)

    def forward(self, dec_input, enc_out, segment_pos, src_mask, trg_mask):
        out = self.decoder(dec_input, enc_out, segment_pos, src_mask, trg_mask)
        return out

dec_input = ['AEAKYAEENCNACCSICPLPNLTISQRIAFIYALYDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKDPEYAVYEIDGLPNLTYAQRRAFIVALWDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKEADAAFAEIFKLPNLTFLQFLAFIQALSDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAKEAESAEKEIVTLPNLTWWQRLAFILALEDDPSQSSELLSEAKKLNDSQAPK',
             'AEAKYAEEPEYAVYEIDGLPNLTYAQRRAFIVALWDDPSQSSELLSEAKKLNDSQAPK']

enc_out = # in training

model = Generative_model()
out = model(dec_input, enc_out, segment_pos, src_mask, trg_mask)

for epoch in range(3):
    running_loss = 0