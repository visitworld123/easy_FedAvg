import copy

class Message(object):
    def __init__(self, content:dict) -> None:
        self.content = content
        self._encode_message()

    def _encode_message(self):
        # TODO
        pass
    
    def decode_message(self):
        # TODO
        decoded_message = copy.deepcopy(self.content)
        return decoded_message

    def get_message(self):
        return self.decode_message()