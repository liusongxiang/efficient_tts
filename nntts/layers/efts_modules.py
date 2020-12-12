import logging
import torch


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ResConv1d(torch.nn.Module):
    """Residual Conv1d layer"""
    def __init__(
        self, 
        n_channels=512,
        k_size=5,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dropout_rate=0.1,
    ):
        super().__init__()
        if dropout_rate < 1e-5:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels, n_channels, 
                    kernel_size=k_size, padding=(k_size-1)//2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(
                    n_channels, n_channels, 
                    kernel_size=k_size, padding=(k_size-1)//2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Dropout(dropout_rate),
            )
        
    def forward(self, x):
        # x [B, C, T]
        x = x + self.conv(x)
        return x


class ResConvBlock(torch.nn.Module):
    """Block containing several ResConv1d layers."""
    def __init__(
        self,
        num_layers,
        n_channels=512,
        k_size=5,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        dropout_rate=0.1,
        use_weight_norm=True,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = torch.nn.Sequential(*[
            ResConv1d(n_channels, k_size, nonlinear_activation, 
                nonlinear_activation_params, dropout_rate) \
            for _ in range(num_layers)
        ])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        # x: [B, C, T]
        return self.layers(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)    


if __name__ == "__main__":
    print("test")
    model = ResConvBlock(5)
    print(model)
    input = torch.rand(2, 512, 80)
    output = model(input)
    print(output.shape)
