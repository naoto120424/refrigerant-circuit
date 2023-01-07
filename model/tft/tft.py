import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, activation=None):
        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = nn.Linear(self.input_size, self.hidden_layer_size)

        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" not in n:
                nn.init.xavier_uniform_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    def forward(self, x):

        if self.dropout_rate:
            x = self.dropout(x)

        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


class GateAddNormNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, activation=None):
        super(GateAddNormNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        self.GLU = GatedLinearUnit(self.input_size, self.hidden_layer_size, self.dropout_rate, activation=self.activation_name)
        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_layer_size, input_size=None, output_size=None, dropout_rate=None, additional_context=None, return_gate=False):
        super(GatedResidualNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate

        self.W1 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = nn.Linear(self.input_size, self.hidden_layer_size)

        if self.additional_context:
            self.W3 = nn.Linear(self.additional_context, self.hidden_layer_size, bias=False)

        if self.output_size:
            self.skip_linear = nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size, self.output_size, self.dropout_rate)
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size, self.hidden_layer_size, self.dropout_rate)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ("W2" in name or "W3" in name) and "bias" not in name:
                nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif ("skip_linear" in name or "W1" in name) and "bias" not in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        if self.additional_context:
            x, context = x
            n2 = F.elu(self.W2(x) + self.W3(context))
        else:
            n2 = F.elu(self.W2(x))

        n1 = self.W1(n2)

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimention

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class InterpretabMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super(InterpretabMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)])
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_layer_size, dropout_rate) -> None:
        super().__init__()
