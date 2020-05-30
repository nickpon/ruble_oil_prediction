import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, t: int):
        """
        Thanks Seanny123 (https://github.com/Seanny123/da-rnn)
        for implementation.

        :param input_size: int
            Number of underlying factors.
        :param hidden_size: int
            Dimension of the hidden state.
        :param t: int
            Number of time steps.
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.t = t

        self.v_e = nn.Linear(in_features=self.t - 1, out_features=1)
        self.W_ehs = nn.Linear(
            in_features=self.hidden_size * 2, out_features=self.t - 1,
        )
        self.U_e = nn.Linear(in_features=self.t - 1, out_features=self.t - 1)
        self.lstm_layer_x_hat = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1,
        )

    def forward(self, input_data: torch.tensor):
        """
        :param input_data: torch.tensor
            Size: [batch_size, T - 1, input_size].
        """
        # [batch_size, T - 1, input_size]
        input_weighted = Variable(
            input_data.data.new(
                input_data.size(0), self.t - 1, self.input_size,
            ).zero_(),
        )
        # [batch_size, T - 1, hidden_size]
        input_encoded = Variable(
            input_data.data.new(
                input_data.size(0), self.t - 1, self.hidden_size,
            ).zero_(),
        )

        # [1, batch_size, hidden_size], [1, batch_size, hidden_size]
        hidden, cell = (
            self.init_hidden(input_data),
            self.init_hidden(input_data),
        )

        for t in range(self.t - 1):
            # [batch_size * input_size, 2 * hidden_size]
            h_cat_s = torch.cat(
                (
                    hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                ),
                dim=2,
            ).reshape(-1, self.hidden_size * 2)

            # [batch_size * input_size, T - 1]
            w_e_dot_h_cat_s = self.W_ehs(h_cat_s)

            # [batch_size * input_size, T - 1]
            u_e_dot_x_k = self.U_e(
                input_data.permute(0, 2, 1).reshape(-1, self.t - 1),
            )

            # [batch_size * input_size, 1]
            v_e = self.v_e(torch.tanh(w_e_dot_h_cat_s + u_e_dot_x_k))

            # [batch_size, input_size]
            v_e_k = v_e.reshape(-1, self.input_size)

            # [batch_size, input_size]
            a_e_k = F.softmax(v_e_k, dim=1)

            # [batch_size, input_size]
            x_hat = torch.mul(a_e_k, input_data[:, t, :])

            self.lstm_layer_x_hat.flatten_parameters()
            _, lstm_x_hat_states = self.lstm_layer_x_hat(
                x_hat.unsqueeze(0), (hidden, cell),
            )

            hidden = lstm_x_hat_states[0]
            cell = lstm_x_hat_states[1]

            input_weighted[:, t, :] = x_hat
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

    def init_hidden(self, x):
        return Variable(x.data.new(1, x.size(0), self.hidden_size).zero_())
