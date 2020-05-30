import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
            self,
            encoder_hidden_size: int,
            decoder_hidden_size: int,
            t: int,
            out_features: int = 1,
    ):
        """
        Thanks Seanny123 (https://github.com/Seanny123/da-rnn)
        for implementation.

        :param encoder_hidden_size: int
            Number of underlying factors of encoder.
        :param decoder_hidden_size: int
            Dimension of the hidden state of decoder.
        :param t: int
            Number of time steps.
        :param out_features: int, [default=1]
            Dimension of the model's output. For union target
            (e.g. min, max. mean) == 1, for multitarget > 1.
        """
        super(Decoder, self).__init__()

        self.t = t
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.out_features = out_features

        self.v_d = nn.Linear(
            in_features=self.encoder_hidden_size, out_features=1,
        )
        self.W_dhs = nn.Linear(
            in_features=2 * self.decoder_hidden_size,
            out_features=encoder_hidden_size,
        )
        self.U_d = nn.Linear(
            in_features=self.encoder_hidden_size,
            out_features=self.encoder_hidden_size,
        )
        self.w = nn.Linear(
            in_features=(1 + self.encoder_hidden_size) * self.out_features,
            out_features=out_features,
        )

        self.w.weight.data.normal_()
        self.lstm_layer = nn.LSTM(
            input_size=out_features, hidden_size=decoder_hidden_size,
        )
        self.fc_final = nn.Linear(
            decoder_hidden_size + out_features * encoder_hidden_size,
            out_features,
        )

    def forward(self, input_encoded: torch.tensor, y_history: torch.tensor):
        """
        :param input_encoded: torch.tensor
            Size: [batch_size,T - 1,encoder_hidden_size].
        :param y_history: torch.tensor
            Size: [batch_size, T - 1].
        """
        # [1, batch_size, decoder_hidden_size]
        hidden = self.init_hidden(input_encoded)
        # [1, batch_size, decoder_hidden_size]
        cell = self.init_hidden(input_encoded)

        # Just not to catch reference before assignment errors.
        context = torch.zeros(1)

        for t in range(self.t - 1):
            # [batch_size, T, 2 * decoder_hidden_size]
            x = torch.cat(
                (
                    hidden.repeat(self.t - 1, 1, 1).permute(1, 0, 2),
                    cell.repeat(self.t - 1, 1, 1).permute(1, 0, 2),
                ),
                dim=2,
            ).reshape(-1, 2 * self.decoder_hidden_size)

            W_dhs_dot_ds = self.W_dhs(x)

            h_i = input_encoded.reshape(-1, self.encoder_hidden_size)

            U_d_dot_h_i = self.U_d(h_i)

            l_i_t = self.v_d(torch.tanh(W_dhs_dot_ds + U_d_dot_h_i))
            lit_t = l_i_t.reshape(-1, self.t - 1)

            b_i_t = F.softmax(lit_t, dim=1)
            bit_t = b_i_t.reshape(-1, self.t - 1)

            bit_t_3d = bit_t.unsqueeze(1)
            context = torch.bmm(bit_t_3d, input_encoded).reshape(
                -1, self.encoder_hidden_size,
            )

            if t < self.t - 1:
                if self.out_features > 1:
                    batch_size, hidden_size = (
                        context.shape[0],
                        context.shape[1],
                    )

                    context = torch.reshape(
                        context.repeat(1, self.out_features),
                        (batch_size, hidden_size, self.out_features),
                    )
                    y_tilde = self.w(
                        torch.cat(
                            (context, y_history[:, t].unsqueeze(1)), dim=1,
                        ).reshape((batch_size, -1)),
                    )
                else:
                    y_tilde = self.w(
                        torch.cat(
                            (context, y_history[:, t].unsqueeze(1)), dim=1,
                        ),
                    )

                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(
                    y_tilde.unsqueeze(0), (hidden, cell),
                )

                # [1, batch_size, decoder_hidden_size]
                hidden = lstm_output[0]
                # [1, batch_size, decoder_hidden_size]
                cell = lstm_output[1]
        if self.out_features > 1:
            y_pred = self.fc_final(
                torch.cat(
                    (hidden[0], context.reshape(context.shape[0], -1)), dim=1,
                ),
            )
        else:
            y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return y_pred

    def init_hidden(self, x):
        return Variable(
            x.data.new(1, x.size(0), self.decoder_hidden_size).zero_(),
        )
