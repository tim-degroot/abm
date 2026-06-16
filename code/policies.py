"""
Policies intercept and modify economic variables.
They are injected into the model at construction.
"""


class NoPolicy:
    """
    Baseline: no policy intervention.
    """

    def on_step_start(self, model):
        """Called at the beginning of each model step."""
        pass

    def modify_credit(self, credit_env):
        """
        Optionally modify credit conditions each step.
        Returns the (possibly modified) CreditEnvironment.
        """
        return credit_env

    def on_transaction(self, transaction, model):
        """
        Called after each ownership transaction completes.
        Can be used to apply transaction taxes, log events, etc.
        """
        pass

    def on_rental_transaction(self, rental_transaction, model):
        """Called after each rental transaction completes."""
        pass
