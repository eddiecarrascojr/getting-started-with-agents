import re

class Guardrail:
    """
    A class to provide safety checks for the AI Agent.
    It validates user inputs and restricts the agent's actions.
    """
    def __init__(self):
        """
        Initializes the Guardrail with predefined rules.
        """
        # A simple list of keywords to block in user input.
        # This can be expanded with more sophisticated checks.
        self.forbidden_words = [
            'delete', 'drop', 'destroy', 'harm', 'illegal', 'password'
        ]

        # A predefined "action space" that limits what the agent is allowed to do.
        # The agent can only call functions that are named in this list.
        self.allowed_actions = [
            'process_csv',
            'quality_check',
            'join_tables',
            'send_message', # Example of another potential safe action
            'fetch_data'    # Example of another potential safe action
        ]
        
        print("Guardrail system initialized.")

    def validate_input(self, user_input: str) -> bool:
        """
        Validates user input by checking for forbidden keywords.

        Args:
            user_input (str): The raw text from the user.

        Returns:
            bool: True if the input is considered safe, False otherwise.
        """
        input_lower = user_input.lower()
        for word in self.forbidden_words:
            # Use regex to match whole words to avoid partial matches (e.g., 'drop' in 'dropbox')
            if re.search(r'\b' + word + r'\b', input_lower):
                print(f"Input rejected by guardrail: Contains forbidden word '{word}'")
                return False
        
        # Add other checks here, e.g., for prompt injection, PII, etc.
        
        return True

    def is_action_allowed(self, action: str) -> bool:
        """
        Checks if the agent's intended action is in the list of allowed actions.

        Args:
            action (str): The name of the function the agent wants to execute.

        Returns:
            bool: True if the action is allowed, False otherwise.
        """
        if action in self.allowed_actions:
            return True
        else:
            print(f"Action '{action}' rejected by guardrail: Not in allowed action space.")
            return False

