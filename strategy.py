class Strategy:
    def __init__(self, leverage, stop_loss, take_profit):
        self.params = {
            "leverage": leverage,
            "sl": stop_loss,
            "tp": take_profit
        }
