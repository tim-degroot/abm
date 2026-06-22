"""
Policies intercept and modify economic variables.
They are injected into the model at construction.
"""

from credit import CreditEnvironment


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

class CreditShockPolicy(NoPolicy): # big hike, seems to be only for households though?
    def on_step_start(self, model):
        if model.steps == 240:
            model.credit = CreditEnvironment(
                mortgage_rate=0.006667,
                ltv_limit=0.80,
                dti_limit=0.30,
                loan_term_months=model.config.credit.loan_term_months,
            )
            print(
                "  [SHOCK] Credit tightened at step 240: "
                "rate=8% p.a. (0.006667/mo), LTV=80%, DTI=30%"
            )
